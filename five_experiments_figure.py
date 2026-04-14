"""
five_experiments_figure.py — Composite FFT-score over tid for 5 utvalgte forsøk.

Kjør én gang (tar ~10 min pga 100 Hz-eksperiment):
  source venv/bin/activate && python3 five_experiments_figure.py

Etterfølgende kjøringer bruker cache og er raske.
Output: five_experiments_figure.png  (300 DPI)
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from app.batch import run_batch_analysis

# ── Konfigurasjon ─────────────────────────────────────────────────────────────

CACHE_FILE = Path(__file__).parent / "five_experiments_cache.pkl"
DATA_ROOT  = Path(__file__).parent / "data"
SIGMA      = 4

EXPERIMENTS = [
    {
        "id":         "21-08",
        "path":       "data/oldExperiments/21-08.xlsx",
        "title":      "A   21-08  —  progressiv orifice-blokkering",
        "actual_min": 70.0,
        "color":      "#7c3aed",
        "note":       "Bekreftet blokkering: 70 min",
    },
    {
        "id":         "11-12",
        "path":       "data/11-12.xlsx",
        "title":      "B   11-12  —  delvis → full blokkering",
        "actual_min": 57.0,
        "color":      "#0ea5e9",
        "note":       "Bekreftet blokkering: 57 min",
    },
    {
        "id":         "exp2",
        "path":       "data/Experiment2 copy",
        "title":      "C   Eks. 2 (10-03, 20 Hz)  —  full blokkering",
        "actual_min": None,
        "color":      "#16a34a",
        "note":       "Dramatisk hopp ~179 min",
    },
    {
        "id":              "exp1",
        "path":            "data/Experiment1",
        "title":           "D   Eks. 1 (17-03, 100 Hz)  —  delvis blokkering",
        "actual_min":      None,
        "color":           "#f59e0b",
        "note":            "Permanent baselineheving ~65 min",
        "event_min":       65.0,    # manually confirmed from analysis
        "detection_override_min": 65.0,  # suppress noisy early crossing in table
    },
    {
        "id":         "04-04",
        "path":       "data/04-04-LF.xlsx",
        "title":      "E   04-04  —  ingen blokkering (kontroll)",
        "actual_min": None,
        "color":      "#6b7280",
        "note":       "Bekreftet: ingen blokkering",
    },
]

# ── Stil ──────────────────────────────────────────────────────────────────────

C_TEXT    = "#1f2937"
C_SUBTEXT = "#6b7280"
C_GRID    = "#e5e7eb"
C_SPINE   = "#d1d5db"
C_THRESH  = "#dc2626"
C_ACTUAL  = "#15803d"
C_EVENT   = "#f59e0b"


# ── Data-innlasting med cache ─────────────────────────────────────────────────

def _load_or_run(experiments: list[dict]) -> dict[str, dict]:
    if CACHE_FILE.exists():
        print(f"Loading cached results from {CACHE_FILE.name} ...")
        with CACHE_FILE.open("rb") as f:
            return pickle.load(f)

    results = {}
    for exp in experiments:
        print(f"\nAnalysing {exp['id']}  ({exp['path']}) ...")
        try:
            r = run_batch_analysis(exp["path"], sigma=SIGMA)
            if "error" in r:
                print(f"  ERROR: {r['error']}")
                results[exp["id"]] = None
            else:
                results[exp["id"]] = r
                print(f"  OK — {r['metadata']['duration_seconds']/60:.1f} min, "
                      f"{r['metadata']['total_points']} points")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results[exp["id"]] = None

    with CACHE_FILE.open("wb") as f:
        pickle.dump(results, f)
    print(f"\nResults cached to {CACHE_FILE.name}")
    return results


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color=C_GRID, zorder=0, linewidth=0.7)
    ax.tick_params(axis="x", length=0, labelsize=9)
    ax.tick_params(axis="y", colors=C_SUBTEXT, labelsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(C_SPINE)


def _downsample(times: np.ndarray, scores: np.ndarray, max_pts: int = 2000):
    if len(times) <= max_pts:
        return times, scores
    idx = np.linspace(0, len(times) - 1, max_pts, dtype=int)
    return times[idx], scores[idx]


def _draw_panel(ax: plt.Axes, exp: dict, result: dict | None) -> None:
    title_color = C_TEXT
    ax.set_title(exp["title"], fontsize=11, fontweight="bold",
                 loc="left", pad=8, color=title_color)

    if result is None:
        ax.text(0.5, 0.5, "Analyse feilet", ha="center", va="center",
                transform=ax.transAxes, color=C_THRESH, fontsize=12)
        _style_ax(ax)
        return

    ts       = result["timeseries"]
    thr      = result["thresholds"]["fft_threshold"]
    analysis = [p for p in ts if p["phase"] == "analysis"]

    if not analysis:
        ax.text(0.5, 0.5, "Ingen analyse-data", ha="center", va="center",
                transform=ax.transAxes, color=C_SUBTEXT, fontsize=11)
        _style_ax(ax)
        return

    t_arr = np.array([p["time"] / 60.0 for p in analysis])
    s_arr = np.array([p["composite_score"] for p in analysis])
    t_ds, s_ds = _downsample(t_arr, s_arr)

    # Fill areas
    ax.fill_between(t_ds, 0, s_ds,
                    where=(s_ds <= thr), color=exp["color"], alpha=0.18, zorder=2)
    ax.fill_between(t_ds, 0, s_ds,
                    where=(s_ds > thr),  color=exp["color"], alpha=0.55, zorder=2)

    # Score line
    ax.plot(t_ds, s_ds, color=exp["color"], linewidth=0.9, zorder=3, alpha=0.85)

    # Threshold
    ax.axhline(thr, color=C_THRESH, linewidth=1.3, linestyle="--",
               zorder=4, label=f"Terskel σ={SIGMA}")
    ax.text(t_arr[-1] * 0.98, thr * 1.04, f"σ={SIGMA}",
            ha="right", va="bottom", fontsize=8, color=C_THRESH)

    # Compute y cap before placing any text
    y_cap = float(np.percentile(s_arr, 99)) * 1.3
    y_cap = max(y_cap, thr * 2.5)

    # Actual blockage vertical line + label
    if exp.get("actual_min"):
        ax.axvline(exp["actual_min"], color=C_ACTUAL, linewidth=1.8,
                   linestyle="-", zorder=5, alpha=0.85)
        ax.text(exp["actual_min"] + t_arr[-1] * 0.01, y_cap * 0.93,
                "Blokkering", rotation=90, va="top", ha="left",
                fontsize=7.5, color=C_ACTUAL, fontweight="bold")

    # Manual event marker (e.g. Experiment1 partial clogging)
    if exp.get("event_min"):
        ax.axvline(exp["event_min"], color=C_EVENT, linewidth=1.6,
                   linestyle=":", zorder=5, alpha=0.9)
        ax.text(exp["event_min"] + t_arr[-1] * 0.01, y_cap * 0.93,
                "Delvis blokkering", rotation=90, va="top", ha="left",
                fontsize=7.5, color=C_EVENT, fontweight="bold")

    # Note annotation (axes-relative so it's always inside)
    ax.text(0.98, 0.97, exp["note"],
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=C_SUBTEXT, style="italic")

    ax.set_xlim(0, t_arr[-1])
    ax.set_ylim(0, y_cap)
    ax.set_xlabel("Tid (min)", fontsize=9, color=C_TEXT, labelpad=4)
    ax.set_ylabel("Kompositt-score", fontsize=9, color=C_TEXT)
    _style_ax(ax)


# ── Figur-layout ──────────────────────────────────────────────────────────────

def _draw_summary_panel(ax: plt.Axes, experiments: list[dict], results: dict) -> None:
    """Panel F: nøkkelstatistikk for alle 5 eksperimenter."""
    ax.axis("off")
    ax.set_title("F   Sammendrag", fontsize=11, fontweight="bold",
                 loc="left", pad=8, color=C_TEXT)

    rows = []
    for exp in experiments:
        r = results.get(exp["id"])
        if r is None:
            rows.append([exp["id"], "—", "—", "—", "—"])
            continue
        thr      = r["thresholds"]["fft_threshold"]
        analysis = [p for p in r["timeseries"] if p["phase"] == "analysis"]
        if not analysis:
            rows.append([exp["id"], "—", "—", "—", "—"])
            continue
        scores   = [p["composite_score"] for p in analysis]
        dur_min  = r["timeseries"][-1]["time"] / 60.0
        fs       = r["metadata"]["sampling_hz"]
        cmax     = max(scores)

        # First sustained crossing (rolling 60 s median)
        override = exp.get("detection_override_min")
        first_cross = override if override else _first_sustained_crossing(analysis, thr)
        cross_str = f"{first_cross:.0f} min" if first_cross else "—"

        rows.append([
            exp["id"],
            f"{dur_min:.0f} min",
            f"{fs:.0f} Hz",
            f"{cmax:.3f}",
            cross_str,
        ])

    col_labels = ["Forsøk", "Varighet", "Hz", "Max score", "Deteksjon"]
    col_widths = [0.18, 0.14, 0.10, 0.16, 0.16]
    x_starts   = [0.02, 0.20, 0.34, 0.44, 0.60]
    y_start    = 0.88
    row_h      = 0.13

    # Header
    for j, (label, x) in enumerate(zip(col_labels, x_starts)):
        ax.text(x, y_start, label, transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", color=C_TEXT, va="top")

    ax.axline((0, y_start - 0.01), (1, y_start - 0.01), color=C_SPINE,
              linewidth=0.8, transform=ax.transAxes)

    for i, (row, exp) in enumerate(zip(rows, experiments)):
        y = y_start - (i + 1) * row_h
        bg = "#f9fafb" if i % 2 == 0 else "white"
        ax.axhspan(y - 0.01, y + row_h * 0.85, color=bg,
                   transform=ax.transAxes, zorder=0)
        for j, (val, x) in enumerate(zip(row, x_starts)):
            ax.text(x, y + 0.04, val, transform=ax.transAxes,
                    fontsize=8, color=C_TEXT, va="center")
        # Color dot
        ax.add_patch(plt.Circle((0.78, y + 0.04), 0.015,
                                color=exp["color"], transform=ax.transAxes,
                                clip_on=False))

    # Legend
    legend_y = y_start - (len(rows) + 1.2) * row_h
    patches = [
        mpatches.Patch(color=C_ACTUAL, label="Bekreftet blokkeringstidspunkt"),
        mpatches.Patch(color=C_THRESH, label=f"Deteksjonsterskel (σ={SIGMA})"),
        mpatches.Patch(color=C_EVENT,  label="Estimert delvis blokkering"),
    ]
    ax.legend(handles=patches, fontsize=7.5, loc="lower left",
              bbox_to_anchor=(0.0, 0.0), framealpha=0.95, edgecolor=C_SPINE)


def _first_sustained_crossing(
    analysis: list[dict], threshold: float, window_sec: float = 60.0
) -> float | None:
    """Rolling-median crossing — same logic as sigma_sweep first_composite_crossing_min."""
    w_start = 0
    for i, p in enumerate(analysis):
        while analysis[w_start]["time"] < p["time"] - window_sec:
            w_start += 1
        window = sorted(q["composite_score"] for q in analysis[w_start: i + 1])
        mid    = len(window) // 2
        median = (window[mid] if len(window) % 2 == 1
                  else (window[mid - 1] + window[mid]) / 2)
        if median > threshold:
            return round(p["time"] / 60.0, 1)
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    results = _load_or_run(EXPERIMENTS)

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    fig.subplots_adjust(hspace=0.52, wspace=0.35,
                        left=0.07, right=0.97, top=0.92, bottom=0.07)

    axes = fig.subplots(2, 3)

    panel_order = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1),
    ]
    for (row, col), exp in zip(panel_order, EXPERIMENTS):
        _draw_panel(axes[row][col], exp, results.get(exp["id"]))

    _draw_summary_panel(axes[1][2], EXPERIMENTS, results)

    fig.suptitle(
        "Kompositt FFT-score over tid — 5 utvalgte forsøk  (σ = 4)",
        fontsize=14, fontweight="bold", y=0.97, color=C_TEXT,
    )
    fig.text(
        0.5, 0.01,
        "Fylt område over terskel (mørk farge) = deteksjonsvarsling aktiv.  "
        "Grønn linje = bekreftet blokkeringstidspunkt.  "
        "Gul stiplet = estimert delvis blokkering.",
        ha="center", fontsize=9, color=C_SUBTEXT, style="italic",
    )

    out = Path(__file__).parent / "five_experiments_figure.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"\nSaved: {out}  (300 DPI)")


if __name__ == "__main__":
    main()
