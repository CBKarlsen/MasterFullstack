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
SIGMAS = [3, 4, 5]
OUTPUT_CSV = Path(__file__).parent / "sigma_sweep_results.csv"
SUPPORTED_EXTENSIONS = {".xlsx", ".csv", ".xls"}

# Subdirectories treated as a single concatenated run (DataStreamer stitches timestamps).
# Each directory contributes one row per sigma, not one row per split file.
EXPERIMENT_DIRS = {"Experiment1", "Experiment2 copy"}
SKIP_FILES = {
    "Book1.xlsx",              # empty
    "clogging results.xlsx",   # non-experiment
    "7-12.xlsx",               # unstable composite across sigmas; excluded from evaluation
    "2108-1.csv",              # 21.08 split 1 — discarded (mid-run spike removed by trimming)
    "01-10.xlsx",              # logger gap mid-file — calibration baseline not valid post-gap
    "11-03-LF.xlsx",           # fails to load in UI — excluded
    "15-03-LF.xlsx",           # fails to load in UI — excluded
    "13-10-LF.xlsx",           # multiple runs + no flow data + messy start — not salvageable without splitting
    "14-07-LF.xlsx",           # multiple runs + no flow data — can't verify first-crossing minute
    "06-12-LF.xlsx",           # multiple runs + no flow data — same as 14-07
    "07-07-LF.xlsx",           # multiple runs + no flow data — same as 14-07
    # 29.06.22 splits — too short for 120 s calibration; use combined 29-06-LF.xlsx instead
    # (Run1 re-included below as no-block ground truth)
    "29-06-LF-Run2.xlsx",
    "2906-2.csv",
    "2906-3.csv",
    "2906-4.csv",
    "30-06-LF.xlsx",           # multiple runs + no flow data
    "3006-3.csv",              # run 3 split — too short for calibration
    "2106-1.csv",              # 21.06 run 1 — too short for calibration
    "2106-2.csv",               # 21.06 run 2 — likely too short; no composite crossing
    "2401-1.csv",               # 24.01 split 1 — only 5 min, too short for calibration
    "2401-2.csv",               # 24.01 split 2 — too short for calibration
    "21-08.xlsx",               # early flow-rate hiccup triggers composite; use 2108-2.csv (clean half) instead
}

# Mirrors the smoothing window used in Batchanalysis.tsx computeCrossings().
# Composite crossing = first point where rolling median over this window exceeds threshold.
# Static uses raw first crossing (no smoothing), matching the UI.
SMOOTHING_WINDOW_SEC = 60.0

# Actual blockage times in minutes from file start, sourced from "Description of cases.docx".
# None = blockage confirmed but exact minute unknown (fill in manually after reviewing UI charts).
# NO_BLOCK = confirmed no blockage (used to flag false positives in the CSV).
NO_BLOCK = -1.0

GROUND_TRUTH: dict[str, float | None] = {
    # 21.08.21 — full file skipped (early hiccup); clean second half labeled below
    "2108-2.csv":         44.0,   # clean second half of 21.08; clog at 44 min from split start

    # 11.12.21 — partial blockage (1500→1200 kg/h) visible at ~57 min in flow chart,
    # then full blockage (1200→500→0 kg/h) near end of file
    "11-12.xlsx":         57.0,

    # 25.03.22 — partial blockage (paper: "thin layer of deposit"). Flow 900→650 kg/h at 2:31.
    # Ice loading on the tank raises baseline spectral variance — composite brushes threshold
    # early (~13 min) before the real event, so first-crossing lead time is inflated.
    "25-03-LF.xlsx":      151.0,

    # 04.04.22 — "No blockages". Last 100 samples trimmed (manual flow shutoff tail).
    "04-04-LF.xlsx":      NO_BLOCK,

    # 27.01.23 — blockage of orifice and loop at 7 Hz, event at 54:44
    "27-01-23-LF.xlsx":   54.733,

    # 18.11.22 — loop blockage at 6 Hz, event at 6:50
    "18-11-LF.xlsx":      6.833,

    # 29.06.22 — 4 runs; blockage in runs 2/3/4 (run 1 = no blockage).
    # Run 1 evaluated separately as a no-block file (composite stays silent).
    "29-06-LF-Run1.xlsx": NO_BLOCK,
    # Combined file: crossing observed at 36:14, corresponds to run 2 blockage.
    "29-06-LF.xlsx":      36.233,

    # 07.09.22 — test section blockage; video 39:34, file clock 29:54
    "0709.csv":           29.9,

    # 21.06.22 — 3 runs: loop blockage (run1), test section (run2), no blockage (run3).
    # All per-run splits skipped (too short for calibration).

    # 24.01.23 — loop blockage. Both splits too short for calibration — all skipped.

    # 17.03 experiment — partial clogging at 1h06m, does not reach full blockage.
    # Concatenated from Experiment1/17-03-high_005.csv through _018.csv.
    "Experiment1":        66.0,

    # 10.03 experiment — clear full blockage at 3h14m.
    # Concatenated from "Experiment2 copy/10-03-high_005.csv" through ~_014.csv.
    "Experiment2 copy":   194.0,
}

FIELDNAMES = [
    # File metadata
    "file", "sigma", "duration_min", "sampling_hz", "smoothing_window_sec",
    # Ground truth
    "actual_blockage_min", "blockage_confirmed",
    # Thresholds
    "fft_threshold", "static_threshold",
    # First threshold crossings
    "composite_first_crossing_min", "static_first_crossing_min",
    # Lead times (actual_blockage - first_crossing; negative = premature / false alarm)
    "composite_lead_min", "static_lead_min",
    # Score statistics
    "composite_max", "static_max",
    "composite_crossings_count", "static_crossings_count",
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
    """Top-level data files + experiment directories (each treated as one virtual file).
    Split files inside experiment dirs are NOT returned individually; DataStreamer
    concatenates them when given the directory path.
    """
    items: list[Path] = []
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_file():
            if (p.suffix.lower() in SUPPORTED_EXTENSIONS
                    and p.name not in SKIP_FILES
                    and not p.name.startswith("~$")):
                items.append(p)
        elif p.is_dir() and p.name in EXPERIMENT_DIRS:
            items.append(p)
    return items


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
    """Raw first crossing — used for static (no smoothing in UI)."""
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
        "composite_first_crossing_min": comp_cross,
        "static_first_crossing_min":   stat_cross,
        "composite_lead_min":          _lead(actual_blockage, comp_cross),
        "static_lead_min":             _lead(actual_blockage, stat_cross),
        "composite_max":               max_score(ts, "composite_score"),
        "static_max":                  max_score(ts, "static_score"),
        "composite_crossings_count":   count_crossings(ts, "composite_score", fft_thr),
        "static_crossings_count":      count_crossings(ts, "static_score", static_thr),
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
