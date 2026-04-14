"""
sigma_analysis.py — Sigma comparison charts + Google Sheets table for thesis.

Run: python3 sigma_analysis.py
Output: sigma_analysis.png, sigma_table.tsv
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Colours ───────────────────────────────────────────────────────────────────

C_COMP     = "#7c3aed"   # purple  — composite method
C_STAT     = "#16a34a"   # green   — static method
C_PREMATURE = "#dc2626"  # red     — premature / false alarm
C_NONE     = "#9ca3af"   # grey    — no detection
SIGMA_COLORS = {3: "#dc2626", 4: "#d97706", 5: "#2563eb"}

SIGMAS = [3, 4, 5]

# ── Experiment data ───────────────────────────────────────────────────────────
# comp/stat values in minutes from file start. None = no threshold crossing.
# actual_min = confirmed blockage time; None = unknown.
# premature = True when detection is known to be too early to be meaningful.

BLOCKAGE_EXPS = [
    {
        "id": "Exp 1",
        "label": "Exp 1\nPartial clog\n(100 Hz, 4h21m)",
        "actual_min": 66.0,
        "sigma": {
            3: {"comp": 10.23,  "stat": None,   "comp_premature": True},
            4: {"comp": 62.98,  "stat": None},
            5: {"comp": 62.98,  "stat": None},
        },
    },
    {
        "id": "Exp 2",
        "label": "Exp 2\nFull clog\n(20 Hz, 3h14m)",
        "actual_min": 194.0,
        "sigma": {
            3: {"comp": 154.53, "stat": 178.40},
            4: {"comp": 154.53, "stat": 179.83},
            5: {"comp": 154.53, "stat": 180.22},
        },
    },
    {
        "id": "11-12",
        "label": "11-12\nPartial→Full\n(1.1 Hz, 4h26m)",
        "actual_min": 57.0,
        "sigma": {
            3: {"comp": 55.673, "stat": 55.14},
            4: {"comp": 55.673, "stat": 55.202},
            5: {"comp": 55.673, "stat": 55.202},
        },
    },
    {
        "id": "21-08",
        "label": "21-08\nProgressive\n(1 Hz, 1h13m)",
        "actual_min": 70.0,
        "sigma": {
            3: {"comp": 5.244,  "stat": 5.478},
            4: {"comp": 5.244,  "stat": 5.544},
            5: {"comp": 5.244,  "stat": 5.644},
        },
    },
    {
        "id": "01-10",
        "label": "01-10\nOrifice block\n(2 Hz, 27m)",
        "actual_min": None,
        "sigma": {
            3: {"comp": 18.316, "stat": 1.258,  "stat_premature": True},
            4: {"comp": 18.316, "stat": 18.416},
            5: {"comp": 18.316, "stat": 18.432},
        },
    },
    {
        "id": "14-07",
        "label": "14-07\n4-run block\n(1 Hz, 1h45m)",
        "actual_min": None,
        "sigma": {
            3: {"comp": 17.3,   "stat": 1.35,   "stat_premature": True},
            4: {"comp": 17.3,   "stat": 1.817,  "stat_premature": True},
            5: {"comp": 17.3,   "stat": 16.9},
        },
    },
    {
        "id": "13-10",
        "label": "13-10\nSudden block\n(1 Hz, 1h33m)",
        "actual_min": None,
        "sigma": {
            3: {"comp": 0.983,  "stat": 0.983},
            4: {"comp": 0.983,  "stat": 0.983},
            5: {"comp": 0.983,  "stat": 0.983},
        },
    },
]

NO_BLOCK_EXPS = [
    {
        "id": "29-06 Run1",
        "label": "29-06 Run1\nNo blockage\n(1 Hz, 41m)",
        "sigma": {
            3: {"comp": None, "stat": None},
            4: {"comp": None, "stat": None},
            5: {"comp": None, "stat": None},
        },
    },
    {
        "id": "04-04",
        "label": "04-04\nNo blockage\n(1 Hz, 5h18m)",
        "sigma": {
            3: {"comp": None, "stat": 58.733},
            4: {"comp": None, "stat": 76.283},
            5: {"comp": None, "stat": 76.617},
        },
    },
]


# ── Panel 1: Composite stability ───────────────────────────────────────────────

def _draw_stability(ax: plt.Axes) -> None:
    """Composite crossing time vs sigma per experiment — should be flat lines."""
    ax.set_facecolor("white")
    for exp in BLOCKAGE_EXPS:
        xs, ys, colors = [], [], []
        for s in SIGMAS:
            v = exp["sigma"][s]
            comp = v.get("comp")
            if comp is None:
                continue
            xs.append(s)
            ys.append(comp)
            colors.append(C_PREMATURE if v.get("comp_premature") else C_COMP)

        if not xs:
            continue
        ax.plot(xs, ys, color=C_COMP, linewidth=1.5, zorder=3, alpha=0.6)
        for x, y, c in zip(xs, ys, colors):
            ax.scatter(x, y, color=c, s=60, zorder=5, edgecolors="white", linewidths=1)

        # Label at σ=4 or last valid point
        lx = xs[-1]
        ly = ys[-1]
        ax.text(lx + 0.06, ly, exp["id"], fontsize=7, va="center", color="#374151")

    ax.set_xticks(SIGMAS)
    ax.set_xticklabels([f"σ={s}" for s in SIGMAS], fontsize=10)
    ax.set_xlabel("Sigma (σ)", fontsize=10, labelpad=6)
    ax.set_ylabel("Detection time (min)", fontsize=10)
    ax.set_title("Composite: crossing time vs σ\n(stable = robust to sigma choice)",
                 fontsize=10, fontweight="bold", loc="left")
    ax.set_xlim(2.6, 5.8)
    _style_ax(ax)


# ── Panel 2: Static sensitivity ───────────────────────────────────────────────

def _draw_sensitivity(ax: plt.Axes) -> None:
    """Static crossing time vs sigma — shows how much sigma changes detection."""
    ax.set_facecolor("white")
    for exp in BLOCKAGE_EXPS:
        xs, ys, colors = [], [], []
        for s in SIGMAS:
            v = exp["sigma"][s]
            stat = v.get("stat")
            if stat is None:
                continue
            xs.append(s)
            ys.append(stat)
            colors.append(C_PREMATURE if v.get("stat_premature") else C_STAT)

        if not xs:
            ax.text(4, 1, f"{exp['id']}: no crossing", fontsize=6,
                    color=C_NONE, ha="center")
            continue
        ax.plot(xs, ys, color=C_STAT, linewidth=1.5, zorder=3, alpha=0.6)
        for x, y, c in zip(xs, ys, colors):
            ax.scatter(x, y, color=c, s=60, zorder=5, edgecolors="white", linewidths=1)

        lx = xs[-1]
        ly = ys[-1]
        ax.text(lx + 0.06, ly, exp["id"], fontsize=7, va="center", color="#374151")

    ax.set_xticks(SIGMAS)
    ax.set_xticklabels([f"σ={s}" for s in SIGMAS], fontsize=10)
    ax.set_xlabel("Sigma (σ)", fontsize=10, labelpad=6)
    ax.set_ylabel("Detection time (min)", fontsize=10)
    ax.set_title("Static: crossing time vs σ\n(steep lines = sigma-sensitive)",
                 fontsize=10, fontweight="bold", loc="left")
    ax.set_xlim(2.6, 5.8)
    _style_ax(ax)


# ── Panel 3: Lead time at σ=4 ─────────────────────────────────────────────────

def _draw_lead_times(ax: plt.Axes) -> None:
    """Advance warning before actual blockage (σ=4 only, known ground truth)."""
    exps_with_gt = [e for e in BLOCKAGE_EXPS if e["actual_min"] is not None]
    labels, comp_leads, stat_leads = [], [], []

    for exp in exps_with_gt:
        v = exp["sigma"][4]
        actual = exp["actual_min"]
        comp = v.get("comp")
        stat = v.get("stat")
        labels.append(exp["id"])
        comp_leads.append(round(actual - comp, 1) if comp is not None else 0)
        stat_leads.append(round(actual - stat, 1) if stat is not None else 0)

    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, comp_leads, w, color=C_COMP, alpha=0.85,
                label="Composite", edgecolor="white")
    b2 = ax.bar(x + w / 2, stat_leads, w, color=C_STAT, alpha=0.85,
                label="Static", edgecolor="white")

    for bar in (*b1, *b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        elif h == 0:
            ax.text(bar.get_x() + bar.get_width() / 2, 1.5,
                    "—", ha="center", va="bottom", fontsize=9, color=C_NONE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Advance warning (min)", fontsize=10)
    ax.set_title("Lead time before blockage (σ=4)\n(confirmed ground truth only)",
                 fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_ylim(0, max(comp_leads + stat_leads) * 1.2 + 5)
    ax.axhline(0, color="#374151", linewidth=0.8)
    _style_ax(ax)


# ── Panel 4: False alarm (no-block runs) ──────────────────────────────────────

def _draw_false_alarms(ax: plt.Axes) -> None:
    """Whether each method fires on confirmed no-block runs (should not fire)."""
    methods = ["Composite", "Static"]
    n_exps = len(NO_BLOCK_EXPS)
    n_sigma = len(SIGMAS)

    cell_w, cell_h = 1.0, 0.8
    for ei, exp in enumerate(NO_BLOCK_EXPS):
        for si, s in enumerate(SIGMAS):
            v = exp["sigma"][s]
            for mi, method_key in enumerate(["comp", "stat"]):
                val = v.get(method_key)
                x = (ei * (n_sigma + 0.5) + si) * cell_w
                y = mi * cell_h
                color = C_PREMATURE if val is not None else "#dcfce7"
                ec = "#dc2626" if val is not None else "#16a34a"
                rect = plt.Rectangle((x, y), cell_w * 0.9, cell_h * 0.85,
                                      facecolor=color, edgecolor=ec,
                                      linewidth=1.5, zorder=2)
                ax.add_patch(rect)
                label = f"{val:.0f}m" if val is not None else "✓"
                fc = "white" if val is not None else "#166534"
                ax.text(x + cell_w * 0.45, y + cell_h * 0.42, label,
                        ha="center", va="center", fontsize=8,
                        fontweight="bold", color=fc)

    # Sigma labels
    for ei, exp in enumerate(NO_BLOCK_EXPS):
        for si, s in enumerate(SIGMAS):
            x = (ei * (n_sigma + 0.5) + si) * cell_w
            ax.text(x + cell_w * 0.45, -0.3, f"σ={s}", ha="center",
                    fontsize=7.5, color="#6b7280")
        mid = (ei * (n_sigma + 0.5) + 1) * cell_w
        ax.text(mid, -0.7, exp["id"], ha="center", fontsize=8.5,
                fontweight="bold", color="#374151")

    # Method labels
    for mi, m in enumerate(methods):
        ax.text(-0.4, mi * cell_h + cell_h * 0.42, m, ha="right",
                va="center", fontsize=8.5, color="#374151")

    fp = mpatches.Patch(facecolor=C_PREMATURE, edgecolor="#dc2626", label="False alarm (fires)")
    tn = mpatches.Patch(facecolor="#dcfce7", edgecolor="#16a34a", label="Correct (no alarm)")
    ax.legend(handles=[tn, fp], fontsize=8, loc="lower right", framealpha=0.9)

    ax.set_xlim(-0.7, (n_exps * (n_sigma + 0.5)) * cell_w)
    ax.set_ylim(-0.9, n_sigma * cell_h)
    ax.set_title("Specificity: no-block runs (should not fire)",
                 fontsize=10, fontweight="bold", loc="left")
    ax.axis("off")


# ── Style helper ──────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes) -> None:
    ax.yaxis.grid(True, color="#f3f4f6", zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", colors="#6b7280", labelsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#d1d5db")


# ── TSV table ─────────────────────────────────────────────────────────────────

def write_tsv_table(path: Path) -> None:
    HEADER = [
        "Experiment", "Type", "Description",
        "Actual blockage (min)", "σ",
        "Composite crossing (min)", "Static crossing (min)",
        "Composite lead (min)", "Static lead (min)",
        "Composite alarm?", "Static alarm?", "Note",
    ]

    def _lead(actual, crossing):
        if actual is None or crossing is None:
            return ""
        return f"{actual - crossing:.1f}"

    def _alarm(crossing):
        return "YES" if crossing is not None else "no"

    rows = [HEADER]

    for exp in BLOCKAGE_EXPS:
        for s in SIGMAS:
            v = exp["sigma"][s]
            comp = v.get("comp")
            stat = v.get("stat")
            actual = exp["actual_min"]
            note = []
            if v.get("comp_premature"):
                note.append("composite premature")
            if v.get("stat_premature"):
                note.append("static premature")
            rows.append([
                exp["id"], "BLOCKAGE", exp["label"].replace("\n", " "),
                actual if actual is not None else "",
                s,
                comp if comp is not None else "",
                stat if stat is not None else "",
                _lead(actual, comp),
                _lead(actual, stat),
                _alarm(comp), _alarm(stat),
                ", ".join(note),
            ])

    for exp in NO_BLOCK_EXPS:
        for s in SIGMAS:
            v = exp["sigma"][s]
            comp = v.get("comp")
            stat = v.get("stat")
            rows.append([
                exp["id"], "NO BLOCK", exp["label"].replace("\n", " "),
                "", s,
                comp if comp is not None else "",
                stat if stat is not None else "",
                "", "",
                _alarm(comp), _alarm(stat),
                "",
            ])

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write("\t".join(str(c) for c in row) + "\n")

    print(f"Table written: {path}  ({len(rows)-1} data rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    fig = plt.figure(figsize=(16, 12), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    _draw_stability(fig.add_subplot(gs[0, 0]))
    _draw_sensitivity(fig.add_subplot(gs[0, 1]))
    _draw_lead_times(fig.add_subplot(gs[1, 0]))
    _draw_false_alarms(fig.add_subplot(gs[1, 1]))

    legend_elements = [
        mpatches.Patch(facecolor=C_COMP, label="Composite method"),
        mpatches.Patch(facecolor=C_STAT, label="Static method"),
        mpatches.Patch(facecolor=C_PREMATURE, label="Premature / false alarm"),
        mpatches.Patch(facecolor=C_NONE, label="No detection"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.95, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Clogging Detection: Sigma Analysis Across 7 Experiments",
        fontsize=14, fontweight="bold", y=0.98,
    )

    out_png = Path(__file__).parent / "sigma_analysis.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Chart saved:  {out_png}")

    write_tsv_table(Path(__file__).parent / "sigma_table.tsv")


if __name__ == "__main__":
    main()
