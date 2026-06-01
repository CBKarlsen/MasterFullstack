"""
Generate three background-chapter figures that illustrate the Composite Method math.

Usage:
    python generate_background_figures.py <data_file> [output_dir]

    data_file:  path to any experiment CSV or XLSX (e.g. data/2106-1.csv)
    output_dir: where to save PNGs (default: current directory)

Figures produced:
    fig1_normalization.png  — why unit-sum normalization removes amplitude bias
    fig2_l1_distance.png    — what the L1 distance measures between two spectra
    fig3_threshold.png      — distribution of calibration scores + percentile threshold
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(__file__))

from app.backend import CloggingDetector
from app.dataloader import DataStreamer
from app.engine import detect_sampling_rate

# ── Style ──────────────────────────────────────────────────────────────────────
# Shared thesis figure style: serif to match the LaTeX body, clean spines,
# and a single export resolution so every figure prints crisply at 300 DPI.
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"
GRAY   = "#6B7280"

# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize(spectrum):
    return spectrum / (np.sum(spectrum) + 1e-10)


def _load_and_calibrate(filepath, sigma=3.0, window_sec=30.0):
    """
    Stream the file, calibrate on the first healthy segment, then collect
    one 'healthy' spectrum and one 'late' spectrum for figure 2.
    Returns (detector, freqs, healthy_raw, late_raw, calib_scores).
    """
    streamer_obj = DataStreamer(filepath)
    streamer = streamer_obj.stream()

    calib_buf, calib_times, all_pts = [], [], []
    for dp in streamer:
        t   = dp.get("time", 0.0)
        dP  = dp.get("p_in", 0.0) - dp.get("p_out", 0.0)
        calib_buf.append(dP)
        calib_times.append(t)
        all_pts.append(dP)
        if len(calib_buf) >= 100:
            fs      = detect_sampling_rate(calib_times)
            needed  = CloggingDetector.required_calibration_samples(fs, window_sec)
            if len(calib_buf) >= needed:
                break

    fs = detect_sampling_rate(calib_times)
    print(f"Detected fs = {fs:.2f} Hz, calibration samples = {len(calib_buf)}")

    detector = CloggingDetector(fs=fs, window_sec=window_sec, sigma=sigma)
    detector.calibrate(calib_buf)

    # Collect all remaining spectra to find a 'late' one
    late_spectrum = None
    max_d, max_spec = -1.0, None
    for dp in streamer:
        dP = dp.get("p_in", 0.0) - dp.get("p_out", 0.0)
        res = detector.process_sample(dP, dp.get("time", 0.0))
        if res and "raw_spectrum" in res:
            raw = np.array(res["raw_spectrum"])
            d   = float(np.sum(np.abs(_normalize(raw) - detector.baseline_spectrum_norm[1:])))
            if d > max_d:
                max_d    = d
                max_spec = raw.copy()

    late_spectrum = max_spec
    freqs = detector.baseline_freqs

    # healthy raw: one of the calibration spectra (first window)
    win = np.array(calib_buf[:detector.window_size])
    ham = win * np.hamming(len(win))
    healthy_raw = np.abs(np.fft.rfft(ham)) ** 2

    return detector, freqs, healthy_raw, late_spectrum


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Normalization removes amplitude bias
# ══════════════════════════════════════════════════════════════════════════════

def fig1_normalization(detector, freqs, healthy_raw, out_path):
    """
    Recompute spectra from detrended signal so the spectral *shape* of
    pressure fluctuations is visible (DC dominates the raw spectrum and
    hides all structure after normalization).
    """
    calib = np.array(detector.baseline_spectrum)   # raw power (with DC)
    window_size = len(calib) * 2 - 2               # rfft inverse: N = 2*(K-1)

    # Build two windows with same shape but 3× amplitude from the calibration
    # baseline by reconstructing via the mean spectrum as a proxy for shape.
    # Detrend: zero out DC bin so spectral fluctuation shape is visible.
    spec_a = calib.copy()
    spec_a[0] = 0.0                  # remove DC for display
    spec_b = spec_a * 3.0            # same shape, 3× amplitude

    norm_a = _normalize(spec_a)
    norm_b = _normalize(spec_b)

    pos = freqs > 0
    f   = freqs[pos]
    sa, sb = spec_a[pos], spec_b[pos]
    na, nb = norm_a[pos], norm_b[pos]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)

    # — Left: raw power spectra look different —
    ax = axes[0]
    ax.semilogy(f, sa, color=BLUE,   lw=1.8, label="Window A")
    ax.semilogy(f, sb, color=ORANGE, lw=1.8, label="Window B  (3× amplitude)")
    ax.set_title("Raw power spectra (DC removed)", fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.legend(frameon=False, fontsize=9)

    # — Right: after normalization they overlap —
    ax = axes[1]
    ax.semilogy(f, na, color=BLUE,   lw=2.5, label="Window A (normalized)")
    ax.semilogy(f, nb, color=ORANGE, lw=1.5, linestyle="--",
                label="Window B (normalized)")
    ax.set_title(r"After unit-sum normalization: $\hat{P}[k]$", fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$\hat{P}[k]$")
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.55, 0.15, "Spectra overlap:\namplitude removed",
            transform=ax.transAxes, color=GREEN, fontsize=9,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREEN, lw=0.8))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — L1 distance between baseline and a clogging-onset spectrum
# ══════════════════════════════════════════════════════════════════════════════

def fig2_l1_distance(detector, freqs, late_raw, out_path):
    # Exclude DC bin so spectral shape differences are visible
    pos      = freqs > 0.05
    pos_any  = freqs > 0          # for computing d_m on full spectrum
    f        = freqs[pos]

    baseline_full = detector.baseline_spectrum_norm[pos_any]
    baseline      = detector.baseline_spectrum_norm[pos]

    if late_raw is not None:
        late_norm_full = _normalize(late_raw)
        # late_raw is already positive-freq (freqs>0), align with pos mask
        pos_in_posfreq = freqs[freqs > 0] > 0.05
        late_norm = late_norm_full[pos_in_posfreq]
    else:
        late_norm = _normalize(np.ones_like(baseline) * baseline.mean())
        late_norm_full = _normalize(np.ones_like(baseline_full) * baseline_full.mean())

    d_m = float(np.sum(np.abs(late_norm_full - baseline_full)))

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(f, baseline,  color=BLUE,   lw=2,   label=r"Baseline $\hat{B}[k]$ (healthy flow)")
    ax.plot(f, late_norm, color=ORANGE, lw=2,   label=r"Current $\hat{P}_m[k]$ (pre-clogging)")

    # Shade the L1 gap
    ax.fill_between(f, baseline, late_norm,
                    where=(late_norm > baseline),
                    alpha=0.25, color=ORANGE, label=r"$|\hat{P}_m - \hat{B}|$ (L1 contribution)")
    ax.fill_between(f, baseline, late_norm,
                    where=(late_norm <= baseline),
                    alpha=0.25, color=BLUE)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Normalized power $\hat{P}[k]$")
    ax.set_title(
        fr"L1 distance $d_m = \sum_k|\hat{{P}}_m[k]-\hat{{B}}[k]| = {d_m:.3f}$  "
        r"(shaded area = per-bin contribution)",
        fontweight="bold"
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Distribution of calibration scores + percentile threshold
# ══════════════════════════════════════════════════════════════════════════════

def fig3_threshold(detector, out_path):
    import math

    scores = detector.baseline_composites
    if len(scores) == 0:
        print("No calibration scores available — skipping figure 3.")
        return

    def sigma_to_pct(sigma):
        return min(99.9, 100.0 * (0.5 * (1.0 + math.erf(sigma / math.sqrt(2.0)))))

    sigmas    = [1, 2, 3, 4]
    colors_s  = ["#86EFAC", "#4ADE80", ORANGE, "#DC2626"]
    thresholds = [float(np.percentile(scores, sigma_to_pct(s))) for s in sigmas]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(scores, bins=40, color=BLUE, alpha=0.7, edgecolor="white", lw=0.4,
            label="Calibration $d_i$ scores")

    for sigma, tau, col in zip(sigmas, thresholds, colors_s):
        ax.axvline(tau, color=col, lw=1.8, linestyle="--",
                   label=fr"$\sigma={sigma}$  →  $\tau={tau:.3f}$  ($p={sigma_to_pct(sigma):.1f}$th pct)")

    ax.set_xlabel(r"Composite score $d_i$")
    ax.set_ylabel("Count")
    ax.set_title(
        r"Distribution of calibration scores and percentile-based thresholds $\tau_\sigma$",
        fontweight="bold"
    )
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Multi-panel results: pressure + composite + static
# ══════════════════════════════════════════════════════════════════════════════

def _smooth(arr, window_pts):
    """Centred rolling median, matching sigma_sweep's composite-crossing smoothing.

    sigma_sweep derives the composite crossing from a rolling *median* over the
    smoothing window (sigma_sweep.SMOOTHING_WINDOW_SEC), so the figure must use the
    same statistic for its first-crossing time to agree with the results tables.
    """
    arr = np.asarray(arr, dtype=float)
    if window_pts < 2:
        return arr.copy()
    # Trailing (causal) window: the real-time UI and sigma_sweep can only see past
    # samples, so a trailing median is what determines the reported crossing time.
    out = np.empty_like(arr)
    for i in range(arr.size):
        out[i] = np.median(arr[max(0, i - window_pts + 1): i + 1])
    return out


def fig4_results_panel(filepath, sigma, out_path,
                       blockage_min=None, window_sec=30.0,
                       smooth_sec=60.0):
    """
    Three vertically stacked panels sharing a time axis:
      1. Raw differential pressure ΔP
      2. Composite score d_m (smoothed) with threshold τ_σ
      3. Static score (smoothed) with its threshold

    Smoothing matches the sigma_sweep smoothing window so the reported
    first-crossing time in the figure agrees with the results tables.
    """
    from app.batch import run_batch_analysis

    # Derive a display name from the filepath, stripping suffixes like " copy"
    exp_name = os.path.splitext(os.path.basename(filepath.rstrip("/")))[0]
    exp_name = exp_name.replace(" copy", "").replace("_copy", "")

    print(f"Running full batch analysis on {filepath} ...")
    result = run_batch_analysis(filepath, sigma=sigma, window_sec=window_sec)

    ts = result["timeseries"]
    comp_thresh = result["thresholds"]["fft_threshold"]
    stat_thresh = result["thresholds"]["static_threshold"]

    # Convert seconds → minutes
    times     = np.array([p["time"] / 60.0 for p in ts])
    pressure  = np.array([p["pressure_drop"] for p in ts])
    composite = np.array([p["composite_score"] for p in ts])
    static    = np.array([p["static_score"] for p in ts])
    phases    = [p["phase"] for p in ts]

    # Estimate sampling rate from time array and smooth
    dt_min = float(np.median(np.diff(times[times > 0]))) if len(times) > 1 else 1/20/60
    smooth_pts = max(1, int(round(smooth_sec / 60.0 / dt_min)))
    composite_smooth = _smooth(composite, smooth_pts)
    static_smooth    = _smooth(static,    smooth_pts)

    # Find calibration boundary
    calib_end_min = next((times[i] for i, ph in enumerate(phases)
                          if ph == "analysis"), None)

    # First sustained crossing on smoothed composite (analysis phase only)
    alarm_min = None
    for i, (ph, d) in enumerate(zip(phases, composite_smooth)):
        if ph == "analysis" and d > comp_thresh:
            alarm_min = times[i]
            break

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # ── Panel 1: raw pressure ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(times, pressure, color=BLUE, lw=0.8, alpha=0.9)
    ax.set_ylabel(r"$\Delta P$ (bar)")
    ax.set_title(f"{exp_name}  —  σ = {sigma}  (smoothed {int(smooth_sec)}s window)",
                 fontweight="bold")

    # ── Panel 2: composite score ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(times, composite, color=BLUE, lw=0.6, alpha=0.3)
    ax.plot(times, composite_smooth, color=BLUE, lw=1.4, alpha=0.95,
            label=r"Composite score $d_m$ (smoothed)")
    ax.axhline(comp_thresh, color=ORANGE, lw=1.8, linestyle="--",
               label=fr"Threshold $\tau_{{\sigma}}={comp_thresh:.4f}$")
    ax.set_ylabel(r"$d_m$")
    ax.set_yscale("log")  # reveal the sustained above-threshold elevation a linear axis hides
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # ── Panel 3: static score ──────────────────────────────────────────────
    ax = axes[2]
    ax.plot(times, static, color=BLUE, lw=0.6, alpha=0.3)
    ax.plot(times, static_smooth, color=BLUE, lw=1.4, alpha=0.95,
            label=r"Static score $|\bar{x}-\mu_0|$ (smoothed)")
    ax.axhline(stat_thresh, color=ORANGE, lw=1.8, linestyle="--",
               label=fr"Threshold $\sigma\cdot s_0={stat_thresh:.5f}$")
    ax.set_ylabel(r"$|\bar{x} - \mu_0|$")
    ax.set_xlabel("Time (minutes)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # ── Shared vertical markers ────────────────────────────────────────────
    for ax in axes:
        if calib_end_min is not None:
            ax.axvspan(times[0], calib_end_min, alpha=0.07,
                       color=GRAY, label="Calibration" if ax is axes[0] else "")

        if alarm_min is not None:
            ax.axvline(alarm_min, color=GREEN, lw=1.8, linestyle="-.",
                       label=f"Composite alarm  ({alarm_min:.1f} min)"
                       if ax is axes[1] else "")

        if blockage_min is not None:
            ax.axvline(blockage_min, color="#DC2626", lw=1.8, linestyle=":",
                       label=f"Confirmed blockage  ({blockage_min:.0f} min)"
                       if ax is axes[1] else "")

    # Add alarm/blockage labels on top panel for readability
    if alarm_min is not None:
        axes[0].axvline(alarm_min, color=GREEN, lw=1.8, linestyle="-.")
        axes[0].text(alarm_min + 1, axes[0].get_ylim()[1] * 0.92,
                     "Alarm", color=GREEN, fontsize=8)
    if blockage_min is not None:
        axes[0].axvline(blockage_min, color="#DC2626", lw=1.8, linestyle=":")
        axes[0].text(blockage_min + 1, axes[0].get_ylim()[1] * 0.80,
                     "Blockage", color="#DC2626", fontsize=8)

    axes[1].legend(frameon=False, fontsize=9, loc="upper left")

    if alarm_min is not None and blockage_min is not None:
        lead = blockage_min - alarm_min
        axes[1].annotate(
            f"Lead time: {lead:.0f} min",
            xy=(alarm_min, comp_thresh),
            xytext=(alarm_min + (blockage_min - alarm_min) * 0.3,
                    comp_thresh * 1.6),
            arrowprops=dict(arrowstyle="->", color=GRAY),
            color=GRAY, fontsize=9
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filepath   = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    blockage_min = float(sys.argv[3]) if len(sys.argv) > 3 else None
    sigma      = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {filepath} ...")
    detector, freqs, healthy_raw, late_raw = _load_and_calibrate(filepath)

    fig1_normalization(
        detector, freqs, healthy_raw,
        os.path.join(output_dir, "fig1_normalization.png")
    )
    fig2_l1_distance(
        detector, freqs, late_raw,
        os.path.join(output_dir, "fig2_l1_distance.png")
    )
    fig3_threshold(
        detector,
        os.path.join(output_dir, "fig3_threshold.png")
    )
    fig4_results_panel(
        filepath, sigma,
        os.path.join(output_dir, "fig4_results_panel.png"),
        blockage_min=blockage_min
    )

    print("\nDone. Four figures written to:", output_dir)
