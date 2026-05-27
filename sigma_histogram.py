"""
sigma_histogram.py — Data-driven sigma comparison from sigma_sweep_results.csv.

Reads the fresh sweep CSV and produces a 3-panel figure answering:
  A. How many files trigger per sigma, broken down by ground truth category?
  B. What is the distribution of composite lead times on confirmed blockages?
  C. When (minutes from file start) do detections happen per sigma?

Run:  python3 sigma_histogram.py
Out:  sigma_histogram.png
Also prints a per-sigma summary table to stdout.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent / "sigma_sweep_results.csv"
OUT_PATH = Path(__file__).parent / "sigma_histogram.png"
SIGMA_COLORS = {3: "#dc2626", 4: "#d97706", 5: "#2563eb"}


# ── Parsing ────────────────────────────────────────────────────────────────

def _to_float(v: str) -> float | None:
    """Parse CSV cell as float, treating empty/None strings as None."""
    if v is None or v.strip() in ("", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_rows() -> list[dict]:
    """Load CSV rows, skipping error rows."""
    with CSV_PATH.open() as fh:
        rows = [r for r in csv.DictReader(fh) if not r.get("error", "").strip()]
    for r in rows:
        r["sigma"] = int(float(r["sigma"]))
        r["composite_cross"] = _to_float(r["composite_first_crossing_min"])
        r["composite_lead"] = _to_float(r["composite_lead_min"])
        r["actual"] = _to_float(r["actual_blockage_min"])
    return rows


# ── Metrics ────────────────────────────────────────────────────────────────

def classify(row: dict) -> str:
    """Classify a single row into a detection outcome category."""
    detected = row["composite_cross"] is not None
    label = row["blockage_confirmed"]
    if label == "yes":
        return "true_positive" if detected else "missed"
    if label == "no_block":
        return "false_positive" if detected else "true_negative"
    return "unknown_detected" if detected else "unknown_silent"


def tally_per_sigma(rows: list[dict]) -> dict[int, dict[str, int]]:
    """Group rows by sigma and count outcome categories."""
    buckets: dict[int, dict[str, int]] = {s: {} for s in SIGMA_COLORS}
    for r in rows:
        if r["sigma"] not in buckets:  # only the sigmas this figure plots (3, 4, 5)
            continue
        cat = classify(r)
        buckets[r["sigma"]][cat] = buckets[r["sigma"]].get(cat, 0) + 1
    return buckets


# ── Plot panels ────────────────────────────────────────────────────────────

def plot_outcome_bars(ax, counts: dict[int, dict[str, int]]) -> None:
    """Panel A — stacked outcome counts per sigma."""
    categories = ["true_positive", "false_positive", "missed",
                  "unknown_detected", "unknown_silent", "true_negative"]
    cat_labels = {
        "true_positive":    "Correct detection (confirmed blockage)",
        "false_positive":   "False positive (no-block)",
        "missed":           "Missed (confirmed blockage)",
        "unknown_detected": "Detection (unknown ground truth)",
        "unknown_silent":   "No detection (unknown ground truth)",
        "true_negative":    "Correctly silent (no-block)",
    }
    cat_colors = {
        "true_positive":    "#16a34a",
        "false_positive":   "#dc2626",
        "missed":           "#f59e0b",
        "unknown_detected": "#7c3aed",
        "unknown_silent":   "#cbd5e1",
        "true_negative":    "#059669",
    }
    sigmas = sorted(counts.keys())
    bottoms = np.zeros(len(sigmas))
    for cat in categories:
        vals = np.array([counts[s].get(cat, 0) for s in sigmas])
        ax.bar(sigmas, vals, bottom=bottoms, color=cat_colors[cat],
               label=cat_labels[cat], edgecolor="white", linewidth=0.5)
        bottoms += vals
    ax.set_xticks(sigmas)
    ax.set_xlabel("Sigma (σ)")
    ax.set_ylabel("Number of files")
    ax.set_title("A. Detection outcomes per σ (Composite)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)


def plot_lead_times(ax, rows: list[dict]) -> None:
    """Panel B — lead times on confirmed blockages, per sigma."""
    for sigma in sorted(SIGMA_COLORS):
        leads = [r["composite_lead"] for r in rows
                 if r["sigma"] == sigma and r["composite_lead"] is not None]
        if not leads:
            continue
        x = [sigma + np.random.uniform(-0.1, 0.1) for _ in leads]
        ax.scatter(x, leads, color=SIGMA_COLORS[sigma], s=80,
                   edgecolors="white", linewidths=1.2, zorder=3,
                   label=f"σ={sigma}  (n={len(leads)}, median={np.median(leads):.1f} min)")
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.set_xticks(sorted(SIGMA_COLORS))
    ax.set_xlabel("Sigma (σ)")
    ax.set_ylabel("Lead time (min); positive = early detection")
    ax.set_title("B. Lead time on confirmed blockages")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)


def plot_crossing_histogram(ax, rows: list[dict]) -> None:
    """Panel C — histogram of composite crossing times, per sigma."""
    all_crosses = [r["composite_cross"] for r in rows if r["composite_cross"] is not None]
    if not all_crosses:
        ax.text(0.5, 0.5, "No detections", transform=ax.transAxes, ha="center")
        return
    bins = np.linspace(0, max(all_crosses) * 1.05, 25)
    for sigma in sorted(SIGMA_COLORS):
        crosses = [r["composite_cross"] for r in rows
                   if r["sigma"] == sigma and r["composite_cross"] is not None]
        ax.hist(crosses, bins=bins, alpha=0.55, color=SIGMA_COLORS[sigma],
                label=f"σ={sigma}  (n={len(crosses)})", edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Detection time (min from file start)")
    ax.set_ylabel("Number of files")
    ax.set_title("C. Distribution of detection times")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)


# ── Summary table ──────────────────────────────────────────────────────────

def print_summary(counts: dict[int, dict[str, int]], rows: list[dict]) -> None:
    """Print a per-sigma markdown summary table to stdout."""
    print("\n=== Sigma summary ===\n")
    print(f"{'σ':>3} {'TP':>4} {'FP':>4} {'Miss':>5} {'Unk+':>5} {'Unk-':>5} "
          f"{'Median lead (min)':>18} {'Mean lead (min)':>17}")
    for sigma in sorted(counts):
        c = counts[sigma]
        leads = [r["composite_lead"] for r in rows
                 if r["sigma"] == sigma and r["composite_lead"] is not None]
        median = f"{np.median(leads):.2f}" if leads else "—"
        mean   = f"{np.mean(leads):.2f}"   if leads else "—"
        print(f"{sigma:>3} "
              f"{c.get('true_positive', 0):>4} "
              f"{c.get('false_positive', 0):>4} "
              f"{c.get('missed', 0):>5} "
              f"{c.get('unknown_detected', 0):>5} "
              f"{c.get('unknown_silent', 0):>5} "
              f"{median:>18} {mean:>17}")
    print("\nTP = true positive (detected on confirmed blockage)")
    print("FP = false positive (detected on no-block file)")
    print("Miss = missed detection on confirmed blockage")
    print("Unk+ = detection on file with unknown ground truth")
    print("Unk- = no detection on file with unknown ground truth")


# ── Entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    rows = load_rows()
    counts = tally_per_sigma(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    plot_outcome_bars(axes[0], counts)
    plot_lead_times(axes[1], rows)
    plot_crossing_histogram(axes[2], rows)
    fig.suptitle(f"Sigma comparison  ({len(rows)} rows from {CSV_PATH.name})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Figure written: {OUT_PATH}")

    print_summary(counts, rows)


if __name__ == "__main__":
    main()
