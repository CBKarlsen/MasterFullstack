"""
exp3_7_figure.py — Publication-quality sigma figure for Experiments 3–7.

Panels:
  A  Composite crossing time vs σ  (flat lines = robust)
  B  Static crossing time vs σ     (steep lines = sigma-sensitive)
  C  Detection time at σ=4 bar chart
  D  Specificity heatmap for no-block runs

Run:  source venv/bin/activate && python3 exp3_7_figure.py
Out:  exp3_7_figure.png  (300 DPI, thesis-ready)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Colours ──────────────────────────────────────────────────────────────────
C_COMP      = "#7c3aed"
C_STAT      = "#16a34a"
C_PREMATURE = "#dc2626"
C_BG_OK     = "#dcfce7"
C_BG_FAIL   = "#fee2e2"
C_TEXT      = "#1f2937"
C_SUBTEXT   = "#6b7280"
C_GRID      = "#e5e7eb"
C_SPINE     = "#d1d5db"

EXP_LINE_COLORS = ["#0ea5e9", "#f59e0b", "#8b5cf6"]
EXP_MARKERS     = ["o", "s", "^"]

SIGMAS = [3, 4, 5]

BLOCKAGE_EXPS = [
    {
        "id": "01-10",
        "label": "01-10  (orifice, 27 min)",
        "sigma": {
            3: {"comp": 18.316, "stat": 1.258,  "stat_premature": True},
            4: {"comp": 18.316, "stat": 18.416},
            5: {"comp": 18.316, "stat": 18.432},
        },
    },
    {
        "id": "14-07",
        "label": "14-07  (loop, 4 runs, 1h45m)",
        "sigma": {
            3: {"comp": 17.3, "stat": 1.35,  "stat_premature": True},
            4: {"comp": 17.3, "stat": 1.817, "stat_premature": True},
            5: {"comp": 17.3, "stat": 16.9},
        },
    },
    {
        "id": "13-10",
        "label": "13-10  (sudden block, 1h33m)",
        "sigma": {
            3: {"comp": 0.983, "stat": 0.983},
            4: {"comp": 0.983, "stat": 0.983},
            5: {"comp": 0.983, "stat": 0.983},
        },
    },
]

NO_BLOCK_EXPS = [
    {
        "id": "29-06 Run1",
        "sub": "41 min",
        "sigma": {
            3: {"comp": None, "stat": None},
            4: {"comp": None, "stat": None},
            5: {"comp": None, "stat": None},
        },
    },
    {
        "id": "04-04",
        "sub": "5h18m",
        "sigma": {
            3: {"comp": None, "stat": 58.733},
            4: {"comp": None, "stat": 76.283},
            5: {"comp": None, "stat": 76.617},
        },
    },
]


# ── Style helper ─────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color=C_GRID, zorder=0, linewidth=0.8)
    ax.tick_params(axis="x", length=0, labelsize=10)
    ax.tick_params(axis="y", colors=C_SUBTEXT, labelsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(C_SPINE)


# ── Panel A: Composite stability ──────────────────────────────────────────────

def _draw_composite_stability(ax: plt.Axes) -> None:
    for i, exp in enumerate(BLOCKAGE_EXPS):
        ys = [exp["sigma"][s]["comp"] for s in SIGMAS]
        ax.plot(SIGMAS, ys, color=EXP_LINE_COLORS[i], marker=EXP_MARKERS[i],
                linewidth=2.2, markersize=8, zorder=3, label=exp["label"],
                markeredgecolor="white", markeredgewidth=1.2)

    ax.set_xticks(SIGMAS)
    ax.set_xticklabels([f"σ = {s}" for s in SIGMAS])
    ax.set_xlabel("Sigma (σ)", fontsize=11, labelpad=6)
    ax.set_ylabel("Detection time (min)", fontsize=11)
    ax.set_title("A   Composite: crossing time vs σ",
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_xlim(2.6, 5.4)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor=C_SPINE,
              loc="center right", handlelength=1.6)
    _style_ax(ax)


# ── Panel B: Static sensitivity ───────────────────────────────────────────────

def _draw_static_sensitivity(ax: plt.Axes) -> None:
    for i, exp in enumerate(BLOCKAGE_EXPS):
        xs, ys, dot_colors = [], [], []
        for s in SIGMAS:
            v = exp["sigma"][s]
            stat = v.get("stat")
            xs.append(s)
            ys.append(stat if stat is not None else np.nan)
            dot_colors.append(C_PREMATURE if v.get("stat_premature") else EXP_LINE_COLORS[i])

        ax.plot(xs, ys, color=EXP_LINE_COLORS[i], linewidth=2.2,
                zorder=3, alpha=0.7, label=exp["label"])
        for x, y, c in zip(xs, ys, dot_colors):
            if not np.isnan(y):
                ax.scatter(x, y, color=c, s=70, zorder=5,
                           edgecolors="white", linewidths=1.2,
                           marker=EXP_MARKERS[i])

    ax.set_xticks(SIGMAS)
    ax.set_xticklabels([f"σ = {s}" for s in SIGMAS])
    ax.set_xlabel("Sigma (σ)", fontsize=11, labelpad=6)
    ax.set_ylabel("Detection time (min)", fontsize=11)
    ax.set_title("B   Static: crossing time vs σ",
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_xlim(2.6, 5.4)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor=C_SPINE,
              loc="upper left", handlelength=1.6)
    _style_ax(ax)


# ── Panel C: Detection time at σ=4 ───────────────────────────────────────────

def _draw_detection_sigma4(ax: plt.Axes) -> None:
    labels      = [e["id"] for e in BLOCKAGE_EXPS]
    comp_vals   = [e["sigma"][4]["comp"] for e in BLOCKAGE_EXPS]
    stat_vals   = [e["sigma"][4].get("stat") for e in BLOCKAGE_EXPS]
    stat_premature = [e["sigma"][4].get("stat_premature", False) for e in BLOCKAGE_EXPS]

    x = np.arange(len(labels))
    w = 0.34

    b_comp = ax.bar(x - w / 2, comp_vals, w, color=C_COMP, alpha=0.85,
                    label="Composite", edgecolor="white", linewidth=0.5, zorder=3)
    stat_heights = [v if v is not None else 0 for v in stat_vals]
    stat_colors  = [C_PREMATURE if p else C_STAT for p in stat_premature]
    b_stat = ax.bar(x + w / 2, stat_heights, w, color=stat_colors, alpha=0.85,
                    label="Static", edgecolor="white", linewidth=0.5, zorder=3)

    for bar in b_comp:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_TEXT)

    for bar, prem in zip(b_stat, stat_premature):
        h = bar.get_height()
        if h > 0:
            suffix = "*" if prem else ""
            clr    = C_PREMATURE if prem else C_TEXT
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}{suffix}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=clr)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Detection time (min)", fontsize=11)
    ax.set_title("C   Detection time at σ=4  (* = premature alarm)",
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.legend(fontsize=9.5, framealpha=0.95, edgecolor=C_SPINE)
    ax.set_ylim(0, max(comp_vals + stat_heights) * 1.25 + 2)
    _style_ax(ax)


# ── Panel D: Specificity heatmap ──────────────────────────────────────────────

def _draw_specificity(ax: plt.Axes) -> None:
    CELL_W, CELL_H = 1.0, 0.95
    GAP_BETWEEN_EXPS = 0.55
    methods     = ["Composite", "Static"]
    method_keys = ["comp", "stat"]

    for ei, exp in enumerate(NO_BLOCK_EXPS):
        x_offset = ei * (len(SIGMAS) + GAP_BETWEEN_EXPS) * CELL_W
        for si, s in enumerate(SIGMAS):
            v = exp["sigma"][s]
            for mi, key in enumerate(method_keys):
                val = v.get(key)
                x   = x_offset + si * CELL_W
                y   = mi * CELL_H
                is_alarm = val is not None
                rect = plt.Rectangle(
                    (x, y), CELL_W * 0.88, CELL_H * 0.84,
                    facecolor=C_BG_FAIL if is_alarm else C_BG_OK,
                    edgecolor=C_PREMATURE if is_alarm else C_STAT,
                    linewidth=1.8, zorder=2,
                )
                ax.add_patch(rect)
                txt = f"{val:.0f} min" if is_alarm else "✓  no alarm"
                fc  = C_PREMATURE if is_alarm else "#166534"
                ax.text(x + CELL_W * 0.44, y + CELL_H * 0.42, txt,
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color=fc)

        # Sigma labels under each column
        for si, s in enumerate(SIGMAS):
            ax.text(x_offset + si * CELL_W + CELL_W * 0.44, -0.22,
                    f"σ={s}", ha="center", fontsize=9, color=C_SUBTEXT)
        # Experiment label
        mid_x = x_offset + CELL_W * 1.0
        ax.text(mid_x, -0.58, f"{exp['id']}  ({exp['sub']})",
                ha="center", fontsize=10, fontweight="bold", color=C_TEXT)

    # Method row labels
    for mi, m in enumerate(methods):
        ax.text(-0.15, mi * CELL_H + CELL_H * 0.42, m,
                ha="right", va="center", fontsize=10,
                fontweight="bold", color=C_TEXT)

    fp = mpatches.Patch(facecolor=C_BG_FAIL, edgecolor=C_PREMATURE, label="False alarm (fires)")
    tn = mpatches.Patch(facecolor=C_BG_OK,   edgecolor=C_STAT,      label="Correct (no alarm)")
    ax.legend(handles=[tn, fp], fontsize=9, loc="upper right",
              framealpha=0.95, edgecolor=C_SPINE,
              bbox_to_anchor=(1.0, 0.95))

    x_max = (len(NO_BLOCK_EXPS) * (len(SIGMAS) + GAP_BETWEEN_EXPS)) * CELL_W
    ax.set_xlim(-1.4, x_max + 1.2)
    ax.set_ylim(-0.75, len(methods) * CELL_H)
    ax.set_title("D   Specificity: no-block runs  (correct = no alarm)",
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.axis("off")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="white")
    fig.subplots_adjust(hspace=0.48, wspace=0.38, top=0.92, bottom=0.08)

    _draw_composite_stability(axes[0, 0])
    _draw_static_sensitivity(axes[0, 1])
    _draw_detection_sigma4(axes[1, 0])
    _draw_specificity(axes[1, 1])

    legend_handles = [
        mpatches.Patch(facecolor=C_COMP,      label="Composite method"),
        mpatches.Patch(facecolor=C_STAT,      label="Static method (on-time)"),
        mpatches.Patch(facecolor=C_PREMATURE, label="Premature / false alarm"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.95, edgecolor=C_SPINE,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "Sigma Analysis — Experiments 3–7\n"
        "Composite stability vs static sensitivity across σ = 3, 4, 5",
        fontsize=13, fontweight="bold", y=0.98, color=C_TEXT,
    )

    out = Path(__file__).parent / "exp3_7_figure.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out}  (300 DPI)")


if __name__ == "__main__":
    main()
