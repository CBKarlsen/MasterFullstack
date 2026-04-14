"""
Sigma sweep — batch-analyses every data file with sigma=3,4,5 and writes results to CSV.
Includes the three forecast regression models (linear, exponential, power-law) and
ground-truth blockage times so the output feeds directly into sigma_performance.py
and detection_timeline.py.

Run: python3 sigma_sweep.py
Output: sigma_sweep_results.csv
"""

import csv
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app.batch import run_batch_analysis

DATA_DIR = Path(__file__).parent / "data"
SIGMAS = [4]   # full 3/4/5 sweep later; single sigma for initial overview
OUTPUT_CSV = Path(__file__).parent / "sigma_sweep_results.csv"
SUPPORTED_EXTENSIONS = {".xlsx", ".csv", ".xls"}
SKIP_FILES = {"Book1.xlsx", "clogging results.xlsx"}   # empty / non-experiment files

# Mirrors the smoothing window used in Batchanalysis.tsx computeCrossings().
# Composite crossing = first point where rolling median over this window exceeds threshold.
# Static and wavelet use raw first crossing (no smoothing), matching the UI.
SMOOTHING_WINDOW_SEC = 60.0

# Actual blockage times in minutes from file start, sourced from "Description of cases.docx".
# None = blockage confirmed but exact minute unknown (fill in manually after reviewing UI charts).
# NO_BLOCK = confirmed no blockage (used to flag false positives in the CSV).
NO_BLOCK = -1.0

GROUND_TRUTH: dict[str, float | None] = {
    # 21.08.21 — "Blockage in 70 minutes" (explicit)
    "21-08.xlsx":         70.0,
    "2108-1.csv":         None,   # split of 21.08 — multiple partial blockages
    "2108-2.csv":         None,   # split of 21.08 — multiple partial blockages

    # 01.10.21 — differential pressure test, orifice blocked at 600 kg/h
    "01-10.xlsx":         None,

    # 07.12.21 — "No Block!"
    "7-12.xlsx":          NO_BLOCK,

    # 11.12.21 — partial blockage (1500→1200 kg/h) visible at ~57 min in flow chart,
    # then full blockage (1200→500→0 kg/h) near end of file
    "11-12.xlsx":         57.0,

    # 11.03.22 — partial blockage, gradual flow reduction
    "11-03-LF.xlsx":      None,

    # 15.03.22 — loop blockage
    "15-03-LF.xlsx":      None,

    # 25.03.22 — thin deposit, explicitly "No Block!"
    "25-03-LF.xlsx":      NO_BLOCK,

    # 04.04.22 — "No blockages"
    "04-04-LF.xlsx":      NO_BLOCK,

    # 13.10.22 — loop blocked at 4.5 Hz
    "13-10-LF.xlsx":      None,

    # 27.01.23 — blockage of orifice and loop at 7 Hz
    "27-01-23-LF.xlsx":   None,

    # 14.07.22 — 4 runs, blockage in all
    "14-07-LF.xlsx":      None,

    # 18.11.22 — loop blockage at 6 Hz
    "18-11-LF.xlsx":      None,

    # 06.12.22 — loop blockage (run 1), then orifice+loop (runs 2 & 3)
    "06-12-LF.xlsx":      None,

    # 07.07.22 — loop blocked outside test section in all 4 runs
    "07-07-LF.xlsx":      None,

    # 29.06.22 — 4 runs; blockage in runs 2/3/4 (run 1 = no blockage)
    "29-06-LF.xlsx":      None,
    "29-06-LF-Run1.xlsx": NO_BLOCK,
    "29-06-LF-Run2.xlsx": None,
    "2906-2.csv":         None,
    "2906-3.csv":         None,
    "2906-4.csv":         None,

    # 30.06.22 — 3 runs; blockage only in run 3
    "30-06-LF.xlsx":      None,
    "3006-3.csv":         None,

    # 07.09.22 — test section blockage at 39:34 on video
    "0709.csv":           None,

    # 21.06.22 — 3 runs: loop blockage (run1), test section (run2), no blockage (run3)
    "2106-1.csv":         None,
    "2106-2.csv":         None,

    # 24.01.23 — loop blockage
    "2401-1.csv":         None,
    "2401-2.csv":         None,
}

FIELDNAMES = [
    # File metadata
    "file", "sigma", "duration_min", "sampling_hz", "smoothing_window_sec",
    # Ground truth
    "actual_blockage_min", "blockage_confirmed",
    # Thresholds
    "fft_threshold", "static_threshold", "wavelet_threshold",
    # First threshold crossings
    "composite_first_crossing_min", "static_first_crossing_min", "wavelet_first_crossing_min",
    # Lead times (actual_blockage - first_crossing; negative = premature / false alarm)
    "composite_lead_min", "static_lead_min",
    # Score statistics
    "composite_max", "static_max", "wavelet_max",
    "composite_crossings_count", "static_crossings_count", "wavelet_crossings_count",
    # Forecast summary
    "forecast_best_fit", "forecast_best_r2",
    "forecast_consensus_eta_min", "forecast_eta_lead_min",
    # Per-model regression results
    "forecast_linear_r2", "forecast_linear_eta_min",
    "forecast_exp_r2", "forecast_exp_eta_min",
    "forecast_power_r2", "forecast_power_eta_min",
    # Flow-based ETA
    "flow_best_fit", "flow_r2", "flow_eta_min", "flow_eta_lead_min",
    "error",
]


def find_data_files() -> list[Path]:
    """Recursively find all data files in DATA_DIR, skipping known non-experiment files."""
    return sorted(
        p for p in DATA_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and p.name not in SKIP_FILES
        and not p.name.startswith("~$")
    )


def first_composite_crossing_min(
    timeseries: list[dict],
    threshold: float,
    smoothing_window_sec: float = SMOOTHING_WINDOW_SEC,
) -> float | None:
    """Rolling-median crossing — mirrors computeCrossings() in Batchanalysis.tsx.
    The composite score is noisy; a transient spike should not count as a detection.
    First crossing = first point where the rolling median over [t-window, t] exceeds threshold.
    """
    pts = [p for p in timeseries if p.get("phase") == "analysis"]
    w_start = 0
    for i, p in enumerate(pts):
        while pts[w_start]["time"] < p["time"] - smoothing_window_sec:
            w_start += 1
        window = sorted(q["composite_score"] for q in pts[w_start : i + 1])
        mid = len(window) // 2
        median = window[mid] if len(window) % 2 == 1 else (window[mid - 1] + window[mid]) / 2
        if median > threshold:
            return round(p["time"] / 60.0, 3)
    return None


def first_raw_crossing_min(timeseries: list[dict], score_key: str, threshold: float) -> float | None:
    """Raw first crossing — used for static and wavelet (no smoothing in UI)."""
    for point in timeseries:
        if point.get("phase") != "analysis":
            continue
        if point.get(score_key, 0.0) > threshold:
            return round(point["time"] / 60.0, 3)
    return None


def count_crossings(timeseries: list[dict], score_key: str, threshold: float) -> int:
    return sum(
        1 for p in timeseries
        if p.get("phase") == "analysis" and p.get(score_key, 0.0) > threshold
    )


def max_score(timeseries: list[dict], score_key: str) -> float:
    values = [p.get(score_key, 0.0) for p in timeseries if p.get("phase") == "analysis"]
    return round(max(values), 6) if values else 0.0


def _lead(actual: float | None, detected: float | None) -> float | None:
    if actual is None or actual == NO_BLOCK or detected is None:
        return None
    return round(actual - detected, 3)


def extract_forecast(forecast: dict | None) -> dict:
    """Pull regression model results out of the forecast dict into flat CSV fields."""
    empty = {
        "forecast_best_fit": "", "forecast_best_r2": "",
        "forecast_consensus_eta_min": "", "forecast_eta_lead_min": "",
        "forecast_linear_r2": "", "forecast_linear_eta_min": "",
        "forecast_exp_r2": "",    "forecast_exp_eta_min": "",
        "forecast_power_r2": "",  "forecast_power_eta_min": "",
    }
    if not forecast:
        return empty

    fits = forecast.get("fits", {})

    def _eta_min(fit_key: str) -> float | None:
        eta_sec = fits.get(fit_key, {}).get("eta_seconds")
        return round(eta_sec / 60.0, 3) if eta_sec is not None else None

    def _r2(fit_key: str) -> float | None:
        val = fits.get(fit_key, {}).get("r2")
        return round(val, 4) if val is not None else None

    best = forecast.get("best_fit", "")
    best_r2 = _r2(best) if best else None
    consensus_eta = forecast.get("consensus_eta")
    consensus_eta_min = round(consensus_eta / 60.0, 3) if consensus_eta is not None else None

    return {
        "forecast_best_fit":          best,
        "forecast_best_r2":           best_r2,
        "forecast_consensus_eta_min": consensus_eta_min,
        "forecast_eta_lead_min":      "",   # filled in extract_row once we have actual_blockage
        "forecast_linear_r2":         _r2("linear"),
        "forecast_linear_eta_min":    _eta_min("linear"),
        "forecast_exp_r2":            _r2("exponential"),
        "forecast_exp_eta_min":       _eta_min("exponential"),
        "forecast_power_r2":          _r2("power_law"),
        "forecast_power_eta_min":     _eta_min("power_law"),
    }


def extract_flow_forecast(flow_forecast: dict | None, actual_blockage: float | None) -> dict:
    empty = {"flow_best_fit": "", "flow_r2": "", "flow_eta_min": "", "flow_eta_lead_min": ""}
    if not flow_forecast:
        return empty
    best = flow_forecast.get("best_fit", "")
    r2 = flow_forecast.get("fits", {}).get(best, {}).get("r2")
    eta_sec = flow_forecast.get("consensus_eta")
    eta_min = round(eta_sec / 60.0, 3) if eta_sec is not None else None
    lead = _lead(actual_blockage, eta_min)
    return {
        "flow_best_fit":    best,
        "flow_r2":          round(r2, 4) if r2 is not None else "",
        "flow_eta_min":     eta_min if eta_min is not None else "",
        "flow_eta_lead_min": lead if lead is not None else "",
    }


def extract_row(filename: str, sigma: float, result: dict) -> dict:
    ts          = result["timeseries"]
    thresholds  = result["thresholds"]
    meta        = result["metadata"]
    duration_min = round(meta["duration_seconds"] / 60.0, 3)

    fft_thr     = thresholds["fft_threshold"]
    static_thr  = thresholds["static_threshold"]
    wavelet_thr = thresholds.get("wavelet_threshold", 0.0)

    # Look up by bare filename so subdirectory paths still match GROUND_TRUTH keys
    bare_name = Path(filename).name
    actual_raw = GROUND_TRUTH.get(bare_name)
    actual_blockage   = None if (actual_raw is None or actual_raw == NO_BLOCK) else actual_raw
    blockage_confirmed = (
        "no_block" if actual_raw == NO_BLOCK
        else ("yes" if actual_raw is not None else "unknown")
    )

    comp_cross  = first_composite_crossing_min(ts, fft_thr)
    stat_cross  = first_raw_crossing_min(ts, "static_score", static_thr)

    forecast_fields = extract_forecast(result.get("forecast"))
    eta_min = forecast_fields["forecast_consensus_eta_min"]
    forecast_fields["forecast_eta_lead_min"] = _lead(actual_blockage, eta_min)

    flow_fields = extract_flow_forecast(result.get("flow_forecast"), actual_blockage)

    return {
        "file":                        filename,
        "sigma":                       sigma,
        "duration_min":                duration_min,
        "sampling_hz":                 meta["sampling_hz"],
        "smoothing_window_sec":        SMOOTHING_WINDOW_SEC,
        "actual_blockage_min":         actual_blockage,
        "blockage_confirmed":          blockage_confirmed,
        "fft_threshold":               round(fft_thr, 6),
        "static_threshold":            round(static_thr, 6),
        "wavelet_threshold":           round(wavelet_thr, 6),
        "composite_first_crossing_min": comp_cross,
        "static_first_crossing_min":   stat_cross,
        "wavelet_first_crossing_min":  first_raw_crossing_min(ts, "wavelet_score", wavelet_thr),
        "composite_lead_min":          _lead(actual_blockage, comp_cross),
        "static_lead_min":             _lead(actual_blockage, stat_cross),
        "composite_max":               max_score(ts, "composite_score"),
        "static_max":                  max_score(ts, "static_score"),
        "wavelet_max":                 max_score(ts, "wavelet_score"),
        "composite_crossings_count":   count_crossings(ts, "composite_score", fft_thr),
        "static_crossings_count":      count_crossings(ts, "static_score", static_thr),
        "wavelet_crossings_count":     count_crossings(ts, "wavelet_score", wavelet_thr),
        **forecast_fields,
        **flow_fields,
        "error": "",
    }


def run_sweep() -> None:
    files = find_data_files()
    if not files:
        print(f"No data files found in {DATA_DIR}")
        return

    print(f"Found {len(files)} file(s). Running sigma sweep {SIGMAS} ...\n")

    rows = []
    for path in files:
        rel_name = str(path.relative_to(DATA_DIR))
        for sigma in SIGMAS:
            print(f"  Processing {rel_name}  σ={sigma} ...", end=" ", flush=True)
            try:
                result = run_batch_analysis(str(path), sigma=sigma)
                if "error" in result:
                    raise ValueError(result["error"])
                row = extract_row(rel_name, sigma, result)
                rows.append(row)
                print(f"OK  (composite: {row['composite_first_crossing_min']} min"
                      f"  ETA: {row['forecast_consensus_eta_min']} min"
                      f"  best: {row['forecast_best_fit']})")
            except Exception as exc:
                print(f"FAILED — {exc}")
                rows.append({
                    "file": rel_name, "sigma": sigma, "error": str(exc),
                    **{k: "" for k in FIELDNAMES if k not in ("file", "sigma", "error")},
                })
                traceback.print_exc()

    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results written to: {OUTPUT_CSV}")
    print(f"  {len(rows)} rows  ({len(files)} files × {len(SIGMAS)} sigmas)")


if __name__ == "__main__":
    run_sweep()
