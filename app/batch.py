# app/batch.py
"""
Batch analysis module for processing entire data files at once.

Unlike the WebSocket streaming engine, this processes all data synchronously
and returns the complete result set. Used for post-hoc analysis where you
want to see the full picture and tweak parameters like sigma.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .backend import CloggingDetector
from .dataloader import DataStreamer


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

def run_batch_analysis(
    filepath: str,
    sigma: float = 3.0,
    calibration_samples: int = 400,
    fs: float = None,  # None = auto-detect
    window_sec: float = 10.0,
) -> Dict[str, Any]:

    streamer_obj = DataStreamer(filepath)
    streamer = streamer_obj.stream()

    # Phase 1: Collect calibration data and detect sampling rate
    calibration_buffer = []
    all_raw_points = []
    columns = []
    detected_fs = None

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
        all_raw_points.append({
            "time": t,
            "flow": flow,
            "pressure_drop": dP,
           "raw": _sanitize(raw_data),
        })

        # Detect fs from first two points
        if detected_fs is None and len(all_raw_points) >= 2:
            dt = all_raw_points[1]["time"] - all_raw_points[0]["time"]
            if dt > 0:
                detected_fs = round(1.0 / dt, 2)

        if len(calibration_buffer) >= calibration_samples:
            break

    # Use detected fs, or fallback
    actual_fs = fs if fs is not None else (detected_fs if detected_fs else 20.0)
    print(f"Batch analysis using fs={actual_fs} Hz")

    # Now create detector with correct fs
    detector = CloggingDetector(fs=actual_fs, window_sec=window_sec, sigma=sigma)

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
        timeseries.append({
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
        })

    # Now process the rest
    for data_point in streamer:
        t = data_point.get("time", 0.0)
        flow = data_point.get("flow", 0.0)
        p_in = data_point.get("p_in", 0.0)
        p_out = data_point.get("p_out", 0.0)
        raw_data = data_point.get("raw", {})
        dP = p_in - p_out

        results = detector.process_sample(dP, t)

        if results:
            timeseries.append({
                "time": t,
                "flow": flow,
                "pressure_drop": dP,
                "static_score": results.get("static", 0.0),
                "composite_score": results.get("composite", 0.0),
                "turbulence_score": results.get("turbulence", 0.0),
                "spectral_slope": results.get("spectral_slope", 0.0),
                "traffic_light": results.get("light_color", "gray"),
                "light_msg": results.get("status_msg", ""),
                "ensemble_probability": results.get("ensemble_probability", 0.0),
                "models": results.get("models", {}),
                "raw": raw_data,
                "phase": "analysis",
            })


    analysis_points = [p for p in timeseries if p["phase"] == "analysis"]
    print(f"Batch complete: {len(timeseries)} total, {len(analysis_points)} analysis points")
    print(f"  Composite range: {min(p['composite_score'] for p in analysis_points) if analysis_points else 'N/A'} - {max(p['composite_score'] for p in analysis_points) if analysis_points else 'N/A'}")
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

    return {
        "timeseries": timeseries_out,
        "columns": columns,
        "calibration": {
            "baseline_mean": detector.baseline_mean,
            "baseline_std": detector.baseline_std,
            "composite_mean": detector.baseline_composite_mean,
            "composite_std": detector.baseline_composite_std,
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
) -> Dict[str, float]:
    """
    Recompute thresholds from stored calibration stats without re-running analysis.

    This allows the frontend to instantly show new threshold lines
    when the user changes sigma.

    Args:
        composite_mean: Baseline composite mean (from calibration response).
        composite_std: Baseline composite std (from calibration response).
        baseline_std: Baseline signal std (from calibration response).
        sigma: New sigma value.

    Returns:
        Dict with new threshold values.
    """
    fft_threshold = composite_mean + sigma * composite_std
    if composite_std <= 0:
        fft_threshold = max(0.05, composite_mean * 5.0)

    static_threshold = sigma * baseline_std

    return {
        "sigma": sigma,
        "fft_threshold": fft_threshold,
        "static_threshold": static_threshold,
    }