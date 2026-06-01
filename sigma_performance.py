"""
Sigma performance chart — detection lead time as a function of the sigma threshold.

Produces a thesis figure (Norwegian labels) comparing how much advance warning each
detection method gives at sigma = 3, 4, 5, for the two clean experiments:
  * Experiment 1 — partial clogging (the composite/FFT method's headline result)
  * Experiment 2 — full clogging

The plotted numbers are NOT recomputed here — they are the summarised detection
lead times (minutes before the ground-truth clogging time) taken from the sigma
sweep runs (see sigma_sweep.py / sigma_sweep_results.csv and the results document).
Each point is the lead time in minutes; an "x" on the axis means the method never
detected at that sigma, and points flagged ``premature`` (the composite σ=3 case
on Experiment 1, which fired ~56 min early) are effectively false alarms and are
drawn in red.

Run:    python3 sigma_performance.py
Output: sigma_performance.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# Lead time = minutes of advance warning before actual clogging
# None = no detection

SIGMAS = [3, 4, 5]

SERIES = [
    {
        "label":  "Kompositt — Eksperiment 1 (delvis tilstopping)",
        "color":  "#2563eb",
        "marker": "o",
        "values": [56, 3, 3],          # σ=3 is premature — marked separately
        "premature": [True, False, False],
    },
    {
        "label":  "Kompositt — Eksperiment 2 (full tilstopping)",
        "color":  "#16a34a",
        "marker": "s",
        "values": [40, 40, 40],
        "premature": [False, False, False],
    },
    {
        "label":  "Statisk — Eksperiment 1 (delvis tilstopping)",
        "color":  "#9ca3af",
        "marker": "^",
        "values": [None, None, None],  # no detection at any sigma
        "premature": [False, False, False],
    },
    {
        "label":  "Statisk — Eksperiment 2 (full tilstopping)",
        "color":  "#7c3aed",
        "marker": "D",
        "values": [16, 15, 14],
        "premature": [False, False, False],
    },
]

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
ax.set_facecolor("white")

for s in SERIES:
    vals = s["values"]
    # Plot line through non-None points
    x_plot = [SIGMAS[i] for i, v in enumerate(vals) if v is not None]
    y_plot = [v           for v in vals              if v is not None]

    if x_plot:
        ax.plot(x_plot, y_plot, color=s["color"], linewidth=2,
                linestyle="--" if all(p for p in s["premature"] if p) else "-",
                zorder=3)

    for i, (sigma, val) in enumerate(zip(SIGMAS, vals)):
        if val is None:
            # No detection marker on x-axis
            ax.plot(sigma, 0, marker="x", color=s["color"],
                    markersize=10, markeredgewidth=2, zorder=4)
            ax.text(sigma, -3.5, "Ingen\ndeteksjon", ha="center",
                    fontsize=7, color=s["color"])
            continue

        is_premature = s["premature"][i]
        marker_color = "#dc2626" if is_premature else s["color"]

        ax.plot(sigma, val, marker=s["marker"], color=marker_color,
                markersize=9, zorder=5,
                markeredgecolor="white", markeredgewidth=1.5)

        # Value label
        offset = 2.5 if not is_premature else 2.5
        ax.text(sigma, val + offset, f"{val} min",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold", color=marker_color)

        # Annotate the premature point
        if is_premature:
            ax.annotate(
                "For tidlig\n(falsk alarm)",
                xy=(sigma, val),
                xytext=(sigma + 0.25, val - 10),
                fontsize=7.5, color="#dc2626",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1),
            )

# Reference line at y=0
ax.axhline(0, color="#374151", linewidth=1, linestyle="-", zorder=1)

# Ideal zone shading (0–10 min = very good)
ax.axhspan(0, 10, alpha=0.06, color="#16a34a", zorder=0)
ax.text(5.08, 5, "Ideell sone\n(< 10 min feil)", fontsize=7.5,
        color="#16a34a", va="center", style="italic")

# Legend
handles = []
for s in SERIES:
    has_data = any(v is not None for v in s["values"])
    ls = "-" if has_data else "-"
    handles.append(plt.Line2D(
        [0], [0], color=s["color"], linewidth=2,
        marker=s["marker"], markersize=7,
        markeredgecolor="white", markeredgewidth=1,
        label=s["label"]
    ))
# Premature marker explanation
handles.append(plt.Line2D(
    [0], [0], marker="o", color="w",
    markerfacecolor="#dc2626", markersize=8,
    label="For tidlig alarm (σ=3, Eks. 1)"
))

ax.legend(handles=handles, fontsize=8, loc="upper right",
          framealpha=0.95, edgecolor="#e5e7eb")

# Axes
ax.set_xticks(SIGMAS)
ax.set_xticklabels([f"σ = {s}" for s in SIGMAS], fontsize=11)
ax.set_xlabel("Sigma-verdi (σ)", fontsize=11, color="#374151", labelpad=8)
ax.set_ylabel("Forvarsel (minutter før faktisk tilstopping)", fontsize=10,
              color="#374151", labelpad=8)
ax.set_xlim(2.5, 5.7)
ax.set_ylim(-10, 70)
ax.set_title(
    "Deteksjonsnøyaktighet per sigma-verdi",
    fontsize=13, fontweight="bold", pad=12,
)

ax.yaxis.grid(True, color="#f3f4f6", zorder=0)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", colors="#6b7280")
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color("#d1d5db")

OUTPUT = "sigma_performance.png"
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"Saved: {OUTPUT}")
