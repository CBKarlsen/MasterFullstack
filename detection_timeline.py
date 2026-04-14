"""
Clogging detection timeline — all sigma levels shown individually.
Run: python3 detection_timeline.py
Output: clogging_detection_timeline.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def hms(h: int, m: int, s: int = 0) -> float:
    return h * 60 + m + s / 60

def fmt(minutes: float) -> str:
    h, m = int(minutes // 60), int(minutes % 60)
    return f"{h:02d}:{m:02d}"


# ── Colours & markers ─────────────────────────────────────────────────────────

C = {3: "#dc2626", 4: "#d97706", 5: "#2563eb", "actual": "#111827"}
COMP_MARKER   = "^"
STATIC_MARKER = "s"


# ── Data ──────────────────────────────────────────────────────────────────────
# Event: (marker_x, ann_x, ann_y, ann_text, legend_label, color, marker)
# marker_x = where to draw the dot on the timeline
# ann_x    = where to anchor the text (can differ to spread overlapping labels)
# ann_y    = positive → above bar, negative → below bar

EXPERIMENTS = [
    {
        "title":       "Experiment 1 — Partial clog at 01:06  (4 h 21 min, 100 Hz)",
        "duration":    261.05,
        "actual":      hms(1, 6, 0),
        "actual_side": "right",
        "note":        "Static method: no threshold crossing at any σ level",
        "events": [
            # Composite σ=3 — premature trigger
            (hms(0,10,14), hms(0,10,14),  0.80,
             "σ=3  00:10\n(+56 min — premature)",
             "Composite  σ=3", C[3], COMP_MARKER),
            # Composite σ=4 — accurate (offset left so labels don't stack)
            (hms(1, 2,59), hms(0,56, 0), -0.58,
             "σ=4  01:03\n(+3 min lead)",
             "Composite  σ=4", C[4], COMP_MARKER),
            # Composite σ=5 — accurate (offset right)
            (hms(1, 2,59), hms(1,10, 0), -0.58,
             "σ=5  01:03\n(+3 min lead)",
             "Composite  σ=5", C[5], COMP_MARKER),
        ],
    },
    {
        "title":       "Experiment 2 — Full clog at 03:14  (3 h 14 min, 20 Hz)",
        "duration":    194.54,
        "actual":      hms(3, 14, 0),
        "actual_side": "left",
        "note":        "Composite crossing time is identical across all σ levels",
        "events": [
            # Composite — all sigma identical, spread markers/labels manually
            (hms(2,34,32), hms(2,22, 0),  0.88,
             "σ=3  02:34\n(+40 min)",
             "Composite  σ=3", C[3], COMP_MARKER),
            (hms(2,34,32), hms(2,33, 0),  0.55,
             "σ=4  02:34\n(+40 min)",
             "Composite  σ=4", C[4], COMP_MARKER),
            (hms(2,34,32), hms(2,44, 0),  0.88,
             "σ=5  02:34\n(+40 min)",
             "Composite  σ=5", C[5], COMP_MARKER),
            # Static — naturally spread across 2 min
            (hms(2,58,24), hms(2,52, 0), -0.55,
             "σ=3  02:58\n(+16 min)",
             "Static  σ=3", C[3], STATIC_MARKER),
            (hms(2,59,50), hms(2,59, 0), -0.88,
             "σ=4  02:59\n(+15 min)",
             "Static  σ=4", C[4], STATIC_MARKER),
            (hms(3, 0,13), hms(3, 5, 0), -0.55,
             "σ=5  03:00\n(+14 min)",
             "Static  σ=5", C[5], STATIC_MARKER),
        ],
    },
]


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_timeline(ax, exp: dict) -> None:
    dur    = exp["duration"]
    actual = exp["actual"]

    ax.set_facecolor("white")

    # Bar background
    ax.broken_barh([(0,      actual)],       (-0.20, 0.40),
                   facecolor="#dcfce7", edgecolor="none", zorder=1)
    ax.broken_barh([(actual, dur - actual)], (-0.20, 0.40),
                   facecolor="#fee2e2", edgecolor="none", zorder=1)
    ax.broken_barh([(0, dur)], (-0.20, 0.40),
                   facecolor="none", edgecolor="#9ca3af", linewidth=0.8, zorder=2)

    # Actual clogging line
    ax.axvline(actual, color=C["actual"], linewidth=2.5, linestyle="--", zorder=10)
    side  = exp.get("actual_side", "right")
    x_off = dur * 0.013
    ax.text(
        actual + (x_off if side == "right" else -x_off), 0.62,
        f"Actual clog\n{fmt(actual)}",
        fontsize=8.5, color=C["actual"], fontweight="bold",
        va="bottom", ha="left" if side == "right" else "right",
    )

    # Events
    legend_handles = []
    seen_legends   = set()
    for marker_x, ann_x, ann_y, ann_text, legend_lbl, color, marker in exp["events"]:
        # Draw marker on the bar
        ax.plot(marker_x, 0, marker=marker, color=color, markersize=10, zorder=8,
                markeredgecolor="white", markeredgewidth=1.5)

        # Arrow tip on the bar edge
        y_tip = 0.20 if ann_y >= 0 else -0.20
        va    = "bottom" if ann_y >= 0 else "top"
        ax.annotate(
            ann_text,
            xy=(marker_x, y_tip),
            xytext=(ann_x, ann_y),
            fontsize=7.8, color=color, ha="center", va=va,
            fontweight="bold", linespacing=1.5,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.7),
        )

        if legend_lbl not in seen_legends:
            seen_legends.add(legend_lbl)
            legend_handles.append(
                plt.Line2D([0], [0], marker=marker, color="w",
                           markerfacecolor=color, markeredgecolor="white",
                           markersize=8, label=legend_lbl)
            )

    legend_handles.append(
        plt.Line2D([0], [0], color=C["actual"], linewidth=2,
                   linestyle="--", label="Actual clogging time")
    )

    # Note
    if "note" in exp:
        ax.text(0, -1.12, exp["note"], fontsize=7.5, color="#6b7280", style="italic")

    # Legend below axes
    ax.legend(
        handles=legend_handles, fontsize=8, ncol=4,
        loc="upper center", bbox_to_anchor=(0.5, -0.24),
        framealpha=0.95, edgecolor="#e5e7eb",
    )

    # Axes
    ax.set_xlim(-dur * 0.025, dur * 1.06)
    ax.set_ylim(-1.28, 1.15)
    ax.set_yticks([])
    ax.set_title(exp["title"], fontsize=10.5, fontweight="bold", loc="left", pad=6)
    ax.set_xlabel("Time into experiment (HH:MM)", fontsize=8.5,
                  color="#6b7280", labelpad=30)

    step  = 30 if dur > 100 else 15
    ticks = np.arange(0, dur + step, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt(t) for t in ticks], fontsize=8)
    ax.tick_params(axis="x", colors="#6b7280", length=3)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")


# ── Lead-time bar chart ───────────────────────────────────────────────────────

def draw_lead_time_chart(ax) -> None:
    data = [
        ("Exp 1\nComposite", 3,  56),
        ("Exp 1\nComposite", 4,   3),
        ("Exp 1\nComposite", 5,   3),
        ("Exp 1\nStatic",    3,   0),
        ("Exp 1\nStatic",    4,   0),
        ("Exp 1\nStatic",    5,   0),
        ("Exp 2\nComposite", 3,  40),
        ("Exp 2\nComposite", 4,  40),
        ("Exp 2\nComposite", 5,  40),
        ("Exp 2\nStatic",    3,  16),
        ("Exp 2\nStatic",    4,  15),
        ("Exp 2\nStatic",    5,  14),
    ]

    groups  = ["Exp 1\nComposite", "Exp 1\nStatic", "Exp 2\nComposite", "Exp 2\nStatic"]
    sigmas  = [3, 4, 5]
    bar_w   = 0.22
    gap     = 0.38
    x       = 0
    centers = []

    for grp in groups:
        c = x + (len(sigmas) - 1) * bar_w / 2
        centers.append(c)
        for i, s in enumerate(sigmas):
            lead = next(v for g, sig, v in data if g == grp and sig == s)
            pos  = x + i * bar_w
            ax.bar(pos, lead, width=bar_w * 0.88, color=C[s],
                   alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)
            if lead > 0:
                ax.text(pos, lead + 0.8, str(lead), ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color=C[s])
            else:
                ax.text(pos, 1.5, "—", ha="center", va="bottom",
                        fontsize=9, color="#9ca3af")
        x += len(sigmas) * bar_w + gap

    # Divider between Exp1 and Exp2
    mid = (centers[1] + centers[2]) / 2
    ax.axvline(mid, color="#e5e7eb", linewidth=1.5, linestyle="--", zorder=1)
    ax.text((centers[0] + centers[1]) / 2, 62, "Experiment 1",
            ha="center", fontsize=9, fontweight="bold", color="#374151")
    ax.text((centers[2] + centers[3]) / 2, 62, "Experiment 2",
            ha="center", fontsize=9, fontweight="bold", color="#374151")

    ax.set_xticks(centers)
    ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("Advance warning (minutes)", fontsize=9, color="#6b7280")
    ax.set_ylim(0, 68)
    ax.set_xlim(-0.3, x - gap + 0.3)
    ax.set_title("Advance Warning by Method and σ Level",
                 fontsize=10.5, fontweight="bold", loc="left", pad=6)
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color="#f3f4f6", zorder=0)
    ax.tick_params(axis="x", length=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#d1d5db")

    handles = [plt.Rectangle((0, 0), 1, 1, color=C[s], label=f"σ={s}") for s in sigmas]
    ax.legend(handles=handles, fontsize=8.5, title="Sigma", title_fontsize=8,
              loc="upper right", framealpha=0.9, edgecolor="#e5e7eb")


# ── Layout ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 13), facecolor="white")
gs  = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.1], hspace=1.05)

draw_timeline(fig.add_subplot(gs[0]), EXPERIMENTS[0])
draw_timeline(fig.add_subplot(gs[1]), EXPERIMENTS[1])
draw_lead_time_chart(fig.add_subplot(gs[2]))

fig.suptitle(
    "Clogging Detection: Composite vs Static Method Across σ Levels",
    fontsize=13, fontweight="bold", y=0.99,
)

OUTPUT = "clogging_detection_timeline.png"
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"Saved: {OUTPUT}")
