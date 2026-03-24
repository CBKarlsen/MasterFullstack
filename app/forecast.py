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
CURVE_POINTS_COUNT = 100
MIN_GROWTH_RATE = 1e-10
MAX_ETA_SECONDS = 30 * 24 * 3600  # beyond 30 days → treat as no crossing


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - float(np.mean(actual))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _fit_linear(t_rel: np.ndarray, scores: np.ndarray, critical: float) -> dict[str, Any]:
    coeffs = np.polyfit(t_rel, scores, 1)
    a, b = float(coeffs[0]), float(coeffs[1])
    r2 = _r_squared(scores, a * t_rel + b)
    eta_rel = float((critical - b) / a) if a > MIN_GROWTH_RATE else None
    return {"r2": r2, "params": {"a": a, "b": b}, "eta_relative": eta_rel}


def _fit_exponential(t_rel: np.ndarray, scores: np.ndarray, critical: float) -> dict[str, Any]:
    valid = scores > 0
    if int(valid.sum()) < MIN_FORECAST_POINTS:
        return {"r2": 0.0, "params": {}, "eta_relative": None}
    log_s, t_v = np.log(scores[valid]), t_rel[valid]
    coeffs = np.polyfit(t_v, log_s, 1)
    k, log_s0 = float(coeffs[0]), float(coeffs[1])
    s0 = float(np.exp(log_s0))
    r2 = _r_squared(log_s, k * t_v + log_s0)
    eta_rel = float(np.log(critical / s0) / k) if k > MIN_GROWTH_RATE and s0 > 0 else None
    return {"r2": r2, "params": {"k": k, "s0": s0}, "eta_relative": eta_rel}


def _fit_power_law(t_rel: np.ndarray, scores: np.ndarray, critical: float) -> dict[str, Any]:
    valid = (scores > 0) & (t_rel > 0)
    if int(valid.sum()) < MIN_FORECAST_POINTS:
        return {"r2": 0.0, "params": {}, "eta_relative": None}
    log_s, log_t = np.log(scores[valid]), np.log(t_rel[valid])
    coeffs = np.polyfit(log_t, log_s, 1)
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


def _find_onset(
    timeseries: list[dict[str, Any]], fft_threshold: float
) -> dict[str, Any] | None:
    return next(
        (
            p for p in timeseries
            if p.get("phase") == "analysis" and p.get("composite_score", 0.0) > fft_threshold
        ),
        None,
    )


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
) -> dict[str, Any] | None:
    """
    Project the time to critical restriction from the post-onset composite score trajectory.

    Fits linear, exponential, and power-law growth models to all composite score data
    after the first threshold crossing, and estimates when the score will reach
    critical_multiplier × fft_threshold.

    Returns None if no threshold crossing exists or post-onset data is insufficient.
    """
    onset = _find_onset(timeseries, fft_threshold)
    if onset is None:
        return None

    onset_time = float(onset["time"])
    critical = fft_threshold * critical_multiplier
    t_rel, scores = _build_post_onset_arrays(timeseries, onset_time)

    if len(scores) < MIN_FORECAST_POINTS:
        return None

    fits: dict[str, dict[str, Any]] = {
        "linear":      _fit_linear(t_rel, scores, critical),
        "exponential": _fit_exponential(t_rel, scores, critical),
        "power_law":   _fit_power_law(t_rel, scores, critical),
    }
    _resolve_absolute_etas(fits, onset_time)

    valid_etas = [f["eta_seconds"] for f in fits.values() if f["eta_seconds"] is not None]
    consensus_eta: float | None = float(np.median(valid_etas)) if valid_etas else None
    best_fit = max(fits.items(), key=lambda kv: kv[1]["r2"])[0]

    return {
        "onset_time": onset_time,
        "detection_threshold": float(fft_threshold),
        "critical_threshold": float(critical),
        "critical_multiplier": float(critical_multiplier),
        "fits": fits,
        "best_fit": best_fit,
        "consensus_eta": consensus_eta,
        "post_onset_points": int(len(scores)),
        "curve_data": {
            name: _generate_curve_points(name, fit.get("params", {}), float(t_rel[-1]), onset_time)
            for name, fit in fits.items()
        },
    }
