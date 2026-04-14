# app/forecast.py
"""
Clogging trajectory forecasting.

Fits three growth models (linear, exponential, power-law) to the composite
score time-series after the first threshold crossing, and projects when the
score will reach a critical level.
"""
import numpy as np
from typing import Any

MIN_FORECAST_POINTS = 20
DEFAULT_CRITICAL_MULTIPLIER = 2.0
DEFAULT_SMOOTHING_WINDOW_SEC = 60.0
DECAY_LAMBDA_FACTOR = 3.0          # weight at onset ≈ exp(-3) ≈ 5% of most-recent
CURVE_POINTS_COUNT = 100
MIN_GROWTH_RATE = 1e-10
MAX_ETA_SECONDS = 30 * 24 * 3600  # beyond 30 days → treat as no crossing
FLOW_ZERO_FRACTION = 0.05          # flow ≤ 5 % of baseline → fully blocked
MIN_FLOW_DECLINE_FRACTION = 0.03   # flow must have dropped ≥ 3 % of baseline to fit


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - float(np.mean(actual))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _smooth_scores_by_time(
    t_rel: np.ndarray, scores: np.ndarray, window_sec: float
) -> np.ndarray:
    """Apply a causal time-based rolling median to remove noise before fitting."""
    smoothed = np.empty_like(scores)
    for i in range(len(scores)):
        in_window = t_rel[: i + 1] >= t_rel[i] - window_sec
        smoothed[i] = float(np.median(scores[: i + 1][in_window]))
    return smoothed


def _build_decay_weights(t_rel: np.ndarray) -> np.ndarray:
    """Exponential decay weights so recent data has more influence on the fit."""
    duration = float(t_rel[-1] - t_rel[0])
    if duration < 1.0:
        return np.ones_like(t_rel, dtype=float)
    lam = DECAY_LAMBDA_FACTOR / duration
    return np.exp(lam * (t_rel - float(t_rel[-1])))


def _fit_linear(
    t_rel: np.ndarray, scores: np.ndarray, critical: float, weights: np.ndarray
) -> dict[str, Any]:
    coeffs = np.polyfit(t_rel, scores, 1, w=weights)
    a, b = float(coeffs[0]), float(coeffs[1])
    r2 = _r_squared(scores, a * t_rel + b)
    eta_rel = float((critical - b) / a) if a > MIN_GROWTH_RATE else None
    return {"r2": r2, "params": {"a": a, "b": b}, "eta_relative": eta_rel}


def _fit_exponential(
    t_rel: np.ndarray, scores: np.ndarray, critical: float, weights: np.ndarray
) -> dict[str, Any]:
    valid = scores > 0
    if int(valid.sum()) < MIN_FORECAST_POINTS:
        return {"r2": 0.0, "params": {}, "eta_relative": None}
    log_s, t_v, w_v = np.log(scores[valid]), t_rel[valid], weights[valid]
    coeffs = np.polyfit(t_v, log_s, 1, w=w_v)
    k, log_s0 = float(coeffs[0]), float(coeffs[1])
    s0 = float(np.exp(log_s0))
    r2 = _r_squared(log_s, k * t_v + log_s0)
    eta_rel = float(np.log(critical / s0) / k) if k > MIN_GROWTH_RATE and s0 > 0 else None
    return {"r2": r2, "params": {"k": k, "s0": s0}, "eta_relative": eta_rel}


def _fit_power_law(
    t_rel: np.ndarray, scores: np.ndarray, critical: float, weights: np.ndarray
) -> dict[str, Any]:
    valid = (scores > 0) & (t_rel > 0)
    if int(valid.sum()) < MIN_FORECAST_POINTS:
        return {"r2": 0.0, "params": {}, "eta_relative": None}
    log_s, log_t, w_v = np.log(scores[valid]), np.log(t_rel[valid]), weights[valid]
    coeffs = np.polyfit(log_t, log_s, 1, w=w_v)
    n, log_a = float(coeffs[0]), float(coeffs[1])
    a = float(np.exp(log_a))
    r2 = _r_squared(log_s, n * log_t + log_a)
    eta_rel = float((critical / a) ** (1.0 / n)) if n > MIN_GROWTH_RATE and a > 0 else None
    return {"r2": r2, "params": {"a": a, "n": n}, "eta_relative": eta_rel}


def _eval_model(model: str, params: dict[str, float], t_vals: np.ndarray) -> np.ndarray:
    if model == "linear":
        return params["a"] * t_vals + params["b"]
    if model == "exponential":
        return params["s0"] * np.exp(params["k"] * t_vals)
    if model == "power_law":
        return params["a"] * np.where(t_vals > 0, t_vals, 1e-9) ** params["n"]
    return np.zeros_like(t_vals)


def _generate_curve_points(
    model: str, params: dict[str, float], t_rel_max: float, onset_time: float
) -> list[dict[str, float]]:
    if not params:
        return []
    t_vals = np.linspace(0.0, t_rel_max * 2.0, CURVE_POINTS_COUNT)
    values = _eval_model(model, params, t_vals)
    return [
        {"time": float(onset_time + t), "value": float(v)}
        for t, v in zip(t_vals, values)
        if np.isfinite(v) and 0.0 <= v <= 10.0
    ]


def _find_onset_time(
    timeseries: list[dict[str, Any]], fft_threshold: float
) -> float | None:
    """Return timestamp of first raw analysis point above threshold, or None."""
    for p in timeseries:
        if p.get("phase") == "analysis" and p.get("composite_score", 0.0) > fft_threshold:
            return float(p["time"])
    return None


def _build_post_onset_arrays(
    timeseries: list[dict[str, Any]], onset_time: float
) -> tuple[np.ndarray, np.ndarray]:
    pts = [
        p for p in timeseries
        if p.get("phase") == "analysis" and p.get("time", 0.0) >= onset_time
    ]
    return (
        np.array([p["time"] for p in pts]) - onset_time,
        np.array([p["composite_score"] for p in pts]),
    )


def _resolve_absolute_etas(fits: dict[str, dict[str, Any]], onset_time: float) -> None:
    """Convert relative ETAs to absolute timestamps in-place."""
    for fit in fits.values():
        eta_rel = fit.pop("eta_relative", None)
        is_valid = eta_rel is not None and 0 < eta_rel < MAX_ETA_SECONDS
        fit["eta_seconds"] = float(onset_time + eta_rel) if is_valid else None


def compute_clogging_forecast(
    timeseries: list[dict[str, Any]],
    fft_threshold: float,
    critical_multiplier: float = DEFAULT_CRITICAL_MULTIPLIER,
    onset_time: float | None = None,
    smoothing_window_sec: float = DEFAULT_SMOOTHING_WINDOW_SEC,
) -> dict[str, Any] | None:
    """
    Project the time to critical restriction from the post-onset composite trajectory.

    onset_time: caller-supplied sustained crossing timestamp (preferred).
                Falls back to first raw sample above threshold when None.
    smoothing_window_sec: rolling median window applied to scores before fitting.
    Returns None if no crossing exists or post-onset data is insufficient.
    """
    resolved_onset = onset_time if onset_time is not None else _find_onset_time(timeseries, fft_threshold)
    if resolved_onset is None:
        return None

    critical = fft_threshold * critical_multiplier
    t_rel, scores = _build_post_onset_arrays(timeseries, resolved_onset)
    if len(scores) < MIN_FORECAST_POINTS:
        return None

    scores = _smooth_scores_by_time(t_rel, scores, smoothing_window_sec)
    weights = _build_decay_weights(t_rel)

    fits: dict[str, dict[str, Any]] = {
        "linear":      _fit_linear(t_rel, scores, critical, weights),
        "exponential": _fit_exponential(t_rel, scores, critical, weights),
        "power_law":   _fit_power_law(t_rel, scores, critical, weights),
    }
    _resolve_absolute_etas(fits, resolved_onset)

    valid_etas = [f["eta_seconds"] for f in fits.values() if f["eta_seconds"] is not None]
    consensus_eta: float | None = float(np.median(valid_etas)) if valid_etas else None
    best_fit = max(fits.items(), key=lambda kv: kv[1]["r2"])[0]

    return {
        "onset_time": resolved_onset,
        "detection_threshold": float(fft_threshold),
        "critical_threshold": float(critical),
        "critical_multiplier": float(critical_multiplier),
        "smoothing_window_sec": float(smoothing_window_sec),
        "fits": fits,
        "best_fit": best_fit,
        "consensus_eta": consensus_eta,
        "post_onset_points": int(len(scores)),
        "curve_data": {
            name: _generate_curve_points(name, fit.get("params", {}), float(t_rel[-1]), resolved_onset)
            for name, fit in fits.items()
        },
    }


# ── Flow-based ETA ────────────────────────────────────────────────────────────

def _calibration_flow_mean(timeseries: list[dict[str, Any]]) -> float:
    """Mean flow rate during calibration phase (baseline reference)."""
    values = [p["flow"] for p in timeseries
              if p.get("phase") == "calibration" and p.get("flow", 0.0) > 0.0]
    return float(np.mean(values)) if values else 0.0


def _post_onset_flow(
    timeseries: list[dict[str, Any]], onset_time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_rel, flow) arrays for all analysis points from onset onward."""
    pts = [p for p in timeseries
           if p.get("phase") == "analysis" and p.get("time", 0.0) >= onset_time]
    return (
        np.array([p["time"] for p in pts], dtype=float) - onset_time,
        np.array([p.get("flow", 0.0) for p in pts], dtype=float),
    )


def _fit_linear_decline(
    t_rel: np.ndarray, flow: np.ndarray, target: float
) -> dict[str, Any]:
    """Linear fit for declining flow — allows negative slope."""
    coeffs = np.polyfit(t_rel, flow, 1)
    a, b = float(coeffs[0]), float(coeffs[1])
    r2 = _r_squared(flow, a * t_rel + b)
    eta_rel = float((target - b) / a) if abs(a) > MIN_GROWTH_RATE else None
    return {"r2": r2, "params": {"a": a, "b": b}, "eta_relative": eta_rel}


def _fit_exp_decay(
    t_rel: np.ndarray, flow: np.ndarray, target: float
) -> dict[str, Any]:
    """Exponential decay fit: flow(t) = f0 * exp(k*t), k < 0."""
    valid = flow > 0
    if int(valid.sum()) < MIN_FORECAST_POINTS:
        return {"r2": 0.0, "params": {}, "eta_relative": None}
    log_f, t_v = np.log(flow[valid]), t_rel[valid]
    coeffs = np.polyfit(t_v, log_f, 1)
    k, log_f0 = float(coeffs[0]), float(coeffs[1])
    f0 = float(np.exp(log_f0))
    r2 = _r_squared(log_f, k * t_v + log_f0)
    can_eta = k < -MIN_GROWTH_RATE and f0 > 0 and target > 0
    eta_rel = float(np.log(target / f0) / k) if can_eta else None
    return {"r2": r2, "params": {"k": k, "f0": f0}, "eta_relative": eta_rel}


def compute_flow_eta(
    timeseries: list[dict[str, Any]], onset_time: float
) -> dict[str, Any] | None:
    """
    Project when flow will reach near-zero using post-onset flow trajectory.

    Returns None when:
    - No flow data in file (baseline ≤ 0)
    - Flow has not declined enough to fit (< MIN_FLOW_DECLINE_FRACTION of baseline)
    - Too few post-onset points
    """
    baseline = _calibration_flow_mean(timeseries)
    if baseline <= 0.0:
        return None

    t_rel, flow = _post_onset_flow(timeseries, onset_time)
    if len(flow) < MIN_FORECAST_POINTS:
        return None

    current_min = float(np.min(flow))
    if baseline - current_min < MIN_FLOW_DECLINE_FRACTION * baseline:
        return None

    target = max(baseline * FLOW_ZERO_FRACTION, 0.0)
    fits: dict[str, Any] = {
        "linear":      _fit_linear_decline(t_rel, flow, target),
        "exponential": _fit_exp_decay(t_rel, flow, target),
    }
    _resolve_absolute_etas(fits, onset_time)

    valid_etas = [f["eta_seconds"] for f in fits.values() if f.get("eta_seconds") is not None]
    consensus_eta: float | None = float(np.median(valid_etas)) if valid_etas else None
    best_fit = max(fits.items(), key=lambda kv: kv[1]["r2"])[0]

    return {
        "baseline_flow": round(baseline, 4),
        "target_flow":   round(target, 4),
        "fits":          fits,
        "best_fit":      best_fit,
        "consensus_eta": consensus_eta,
    }
