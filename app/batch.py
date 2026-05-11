# app/batch.py
"""
Batch analysis module for processing entire data files at once.

Unlike the WebSocket streaming engine, this processes all data synchronously
and returns the complete result set. Used for post-hoc analysis where you
want to see the full picture and tweak parameters like sigma.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .backend import CloggingDetector, _sigma_to_pct
from .dataloader import DataStreamer
from .engine import detect_sampling_rate
from .forecast import compute_clogging_forecast, compute_flow_eta
from .models import get_registry


def _sanitize(value):
    """Replace NaN/Inf with 0.0 for JSON safety."""
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return 0.0
    if isinstance(value, np.floating):
        v = float(value)
        return 0.0 if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def _apply_batch_ml(
    timeseries: List[Dict[str, Any]], features_list: List[Dict[str, float]]
) -> None:
    """
    Run ML inference once over all analysis points in one vectorised call per model.

    Models that expose ``predict_batch()`` (e.g. sklearn RF) process the full
    feature matrix in a single call — orders of magnitude faster than calling
    ``predict()`` 200k+ times individually.  Other models fall back to a plain
    Python loop (still skipping the per-sample overhead from process_sample).
    """
    if not features_list:
        return

    registry = get_registry()
    active_models = registry.get_active()
    if not active_models:
        return

    # Build per-model probability arrays
    n = len(features_list)
    model_pred_rows: Dict[str, List[Dict[str, Any]]] = {}

    for name, model in active_models.items():
        try:
            if hasattr(model, "predict_batch") and callable(model.predict_batch):
                # Vectorized path (e.g. sklearn RandomForest)
                batch_results = model.predict_batch(features_list)
                model_pred_rows[name] = [
                    {**r.to_dict(), "weight": model.metadata.weight}
                    for r in batch_results
                ]
            else:
                # Sequential fallback for models that don't support batching
                model_pred_rows[name] = [
                    {**model.predict(f).to_dict(), "weight": model.metadata.weight}
                    for f in features_list
                ]
        except Exception as exc:
            model_pred_rows[name] = [
                {
                    "probability": 0.0,
                    "confidence": 0.0,
                    "weight": model.metadata.weight,
                    "error": str(exc),
                }
            ] * n

    # Write ML results back into timeseries analysis points
    analysis_indices = [
        i for i, p in enumerate(timeseries) if p.get("phase") == "analysis"
    ]

    for j, ts_idx in enumerate(analysis_indices):
        if j >= n:
            break

        models_at_j: Dict[str, Any] = {
            name: preds[j] for name, preds in model_pred_rows.items()
        }

        # Weighted ensemble (mirrors CloggingDetector._calculate_ensemble)
        probs, weights = [], []
        for r in models_at_j.values():
            prob = r.get("probability", 0.0)
            conf = r.get("confidence", 1.0)
            wt = r.get("weight", 1.0)
            if "error" not in r and prob > 0:
                probs.append(prob)
                weights.append(conf * wt)

        if probs:
            total = sum(weights)
            ensemble = (
                sum(p * w for p, w in zip(probs, weights)) / total
                if total > 0
                else float(np.mean(probs))
            )
        else:
            ensemble = 0.0

        timeseries[ts_idx]["models"] = models_at_j
        timeseries[ts_idx]["ensemble_probability"] = ensemble


def run_batch_analysis(
    filepath: str,
    sigma: float = 3.0,
    calibration_seconds: Optional[float] = None,
    fs: Optional[float] = None,  # None = auto-detect
    window_sec: float = 30.0,
) -> Dict[str, Any]:
    """Run batch analysis.

    Args:
        calibration_seconds: If None, uses the same dynamic sample count as the
            real-time engine — ``CloggingDetector.required_calibration_samples(fs)``
            — so both modes produce identical scores. Pass an explicit value only
            to override that for sensitivity studies.
        fs: If None, fs is detected from timestamp deltas using the same
            median-of-deltas method as the real-time engine.
    """

    streamer_obj = DataStreamer(filepath)
    streamer = streamer_obj.stream()

    # Phase 1: collect calibration data. Use the same sample count as real-time
    # by default so both modes produce identical scores for the same file.
    calibration_buffer: List[float] = []
    calibration_times: List[float] = []
    all_raw_points: List[Dict[str, Any]] = []
    columns: List[str] = []

    for data_point in streamer:
        t = data_point.get("time", 0.0)
        flow = data_point.get("flow", 0.0)
        p_in = data_point.get("p_in", 0.0)
        p_out = data_point.get("p_out", 0.0)
        raw_data = data_point.get("raw", {})
        dP = p_in - p_out

        if not columns:
            columns = data_point.get("columns", [])

        calibration_buffer.append(dP)
        calibration_times.append(t)
        all_raw_points.append({
            "time": t,
            "flow": flow,
            "pressure_drop": dP,
            "raw": _sanitize(raw_data),
        })

        # We need fs before we know how many samples to collect. Once we have
        # enough timestamps to estimate fs, size the buffer to match real-time:
        #   - default: CloggingDetector.required_calibration_samples(fs)
        #   - override: calibration_seconds * fs (for sensitivity studies)
        if len(calibration_buffer) >= 100:
            tentative_fs = fs if fs is not None else detect_sampling_rate(calibration_times)
            if calibration_seconds is None:
                needed = CloggingDetector.required_calibration_samples(tentative_fs, window_sec)
            else:
                needed = max(int(calibration_seconds * tentative_fs), 50)
            if len(calibration_buffer) >= needed:
                break

    actual_fs = fs if fs is not None else detect_sampling_rate(calibration_times)
    calibration_samples = len(calibration_buffer)
    print(f"Batch analysis using fs={actual_fs} Hz, calibration={calibration_samples} samples")

    # Now create detector with correct fs.
    # enable_models=False skips per-sample ML inference; we batch-predict at the end.
    detector = CloggingDetector(
        fs=actual_fs, window_sec=window_sec, sigma=sigma, enable_models=False
    )

    if len(calibration_buffer) < 50:
        return {
            "error": f"Not enough data. Got {len(calibration_buffer)}, need at least 50.",
            "timeseries": [],
        }

    detector.calibrate(calibration_buffer)

    # Phase 2: Process all remaining data
    timeseries = []

    # First, add calibration-phase points (no scores yet)
    for pt in all_raw_points:
        timeseries.append(
            {
                "time": pt["time"],
                "flow": pt["flow"],
                "pressure_drop": pt["pressure_drop"],
                "static_score": 0.0,
                "composite_score": 0.0,
                "turbulence_score": 0.0,
                "spectral_slope": 0.0,
                "traffic_light": "gray",
                "light_msg": "Calibrating",
                "ensemble_probability": 0.0,
                "raw": pt["raw"],
                "phase": "calibration",
            }
        )

    # Now process the rest — ML is disabled per-sample, features collected for batch inference
    collected_features: List[
        Dict[str, float]
    ] = []  # parallel to analysis entries in timeseries

    for data_point in streamer:
        t = data_point.get("time", 0.0)
        flow = data_point.get("flow", 0.0)
        p_in = data_point.get("p_in", 0.0)
        p_out = data_point.get("p_out", 0.0)
        raw_data = data_point.get("raw", {})
        dP = p_in - p_out

        results = detector.process_sample(dP, t)

        if results:
            timeseries.append(
                {
                    "time": t,
                    "flow": flow,
                    "pressure_drop": dP,
                    "static_score": results.get("static", 0.0),
                    "composite_score": results.get("composite", 0.0),
                    "turbulence_score": results.get("turbulence", 0.0),
                    "spectral_slope": results.get("spectral_slope", 0.0),
                    "traffic_light": results.get("light_color", "gray"),
                    "light_msg": results.get("status_msg", ""),
                    "ensemble_probability": 0.0,
                    "models": {},
                    "raw": _sanitize(raw_data),
                    "phase": "analysis",
                }
            )
            collected_features.append(
                results.get(
                    "_features",
                    {
                        "static_score": results.get("static", 0.0),
                        "composite_score": results.get("composite", 0.0),
                        "turbulence_score": results.get("turbulence", 0.0),
                        "spectral_slope": results.get("spectral_slope", 0.0),
                    },
                )
            )

    analysis_points = [p for p in timeseries if p["phase"] == "analysis"]
    print(
        f"Batch complete: {len(timeseries)} total, {len(analysis_points)} analysis points"
    )

    # --- Batch ML inference (single vectorized call per model, much faster) ---
    _apply_batch_ml(timeseries, collected_features)
    print(
        f"  Composite range: {min(p['composite_score'] for p in analysis_points) if analysis_points else 'N/A'} - {max(p['composite_score'] for p in analysis_points) if analysis_points else 'N/A'}"
    )
    # Build response
    duration = timeseries[-1]["time"] if timeseries else 0.0

    max_points = 5000
    if len(timeseries) > max_points:
        step = len(timeseries) / max_points
        downsampled = []
        for i in range(max_points):
            idx = int(i * step)
            downsampled.append(timeseries[idx])
        # Always include the last point
        downsampled[-1] = timeseries[-1]
        timeseries_out = downsampled
    else:
        timeseries_out = timeseries

    forecast = compute_clogging_forecast(timeseries_out, detector.fft_threshold)

    flow_forecast = None
    if forecast is not None:
        flow_forecast = compute_flow_eta(timeseries_out, forecast["onset_time"])

    return {
        "timeseries": timeseries_out,
        "forecast": forecast,
        "flow_forecast": flow_forecast,
        "columns": columns,
        "calibration": {
            "baseline_mean": detector.baseline_mean,
            "baseline_std": detector.baseline_std,
            "composite_mean": detector.baseline_composite_mean,
            "composite_std": detector.baseline_composite_std,
            "composite_p99": float(np.percentile(detector.baseline_composites, 99))
            if len(detector.baseline_composites) > 0
            else None,
            "samples_used": calibration_samples,
        },
        "thresholds": {
            "sigma": sigma,
            "fft_threshold": detector.fft_threshold,
            "static_threshold": detector.critical_threshold,
        },
        "metadata": {
            "total_points": len(timeseries),
            "duration_seconds": duration,
            "sampling_hz": actual_fs,
            "file": filepath,
        },
    }


def recompute_thresholds(
    composite_mean: float,
    composite_std: float,
    baseline_std: float,
    sigma: float,
    composite_p99: float = None,
) -> Dict[str, float]:
    """
    Recompute thresholds from stored calibration stats without re-running analysis.

    Uses percentile-based threshold for composite (matching CloggingDetector),
    falling back to mean+sigma*std only when the percentile is unavailable.

    Args:
        composite_mean: Baseline composite mean (from calibration response).
        composite_std: Baseline composite std (from calibration response).
        baseline_std: Baseline signal std (from calibration response).
        sigma: New sigma value.
        composite_p99: 99th-percentile composite from calibration (preferred).

    Returns:
        Dict with new threshold values.
    """
    if composite_p99 is not None:
        pct = _sigma_to_pct(sigma)
        fft_threshold = composite_mean + (composite_p99 - composite_mean) * (pct / 99.0)
    elif composite_std > 0:
        fft_threshold = composite_mean + sigma * composite_std
    else:
        fft_threshold = max(0.05, composite_mean * (1.0 + sigma * 0.1))

    static_threshold = sigma * baseline_std

    return {
        "sigma": sigma,
        "fft_threshold": fft_threshold,
        "static_threshold": static_threshold,
    }
