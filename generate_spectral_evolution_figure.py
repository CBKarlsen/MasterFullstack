"""
generate_spectral_evolution_figure.py

Produces two thesis figures illustrating the Composite Method directly
on Experiment2 data (fs = 20 Hz, confirmed blockage at 194 min).

Outputs (saved in the script's directory):
  fig_spectrogram_exp2.png        — STFT spectrogram + Composite score timeline
  fig_spectral_snapshots_exp2.png — 3-panel normalized spectrum comparison

Usage:
    source venv/bin/activate
    python generate_spectral_evolution_figure.py
"""

import sys
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app.backend import CloggingDetector, CALIBRATION_STEP_RATIO, CALIBRATION_MIN_WINDOWS
from app.dataloader import DataStreamer
from app.engine import detect_sampling_rate

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH              = Path("data/Experiment2 copy")
SIGMA                  = 3.0
WINDOW_SEC             = 30.0      # 600 samples at 20 Hz — default in CloggingDetector
CONFIRMED_BLOCKAGE_MIN = 194.0     # ground-truth label from thesis Table 5.1

OUT_SPECTROGRAM  = Path("fig_spectrogram_exp2.png")
OUT_SNAPSHOTS    = Path("fig_spectral_snapshots_exp2.png")

# ── Thesis plot style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"
GRAY   = "#6B7280"
RED    = "#DC2626"
PURPLE = "#7C3AED"

# ── Helpers ────────────────────────────────────────────────────────────────────

def hamming_fft_norm(signal: np.ndarray, fs: float):
    """
    One-sided normalized power spectrum — implements Equations 2.4–2.5.
    Returns (freqs, P_raw, P_norm).
    """
    w = np.hamming(len(signal))
    X = np.fft.rfft(signal * w)
    P = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    P_norm = P / (np.sum(P) + 1e-10)
    return freqs, P, P_norm


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance between two unit-sum spectra — Equation 2.7."""
    return float(np.sum(np.abs(a - b)))


# ── Step 1: load all pressure samples ─────────────────────────────────────────

print(f"Loading {DATA_PATH} ...")
streamer = DataStreamer(str(DATA_PATH))

times_raw, pressures = [], []
for dp in streamer.stream():
    times_raw.append(dp.get("time", 0.0))
    pressures.append(dp.get("p_in", 0.0) - dp.get("p_out", 0.0))

pressures = np.asarray(pressures, dtype=np.float64)
times_raw = np.asarray(times_raw, dtype=np.float64)
# Re-zero time so t=0 is start of recording
times_sec = times_raw - times_raw[0]
times_min = times_sec / 60.0

fs = float(np.round(1.0 / np.median(np.diff(times_sec[: min(500, len(times_sec))]))))
print(f"  {len(pressures):,} samples  |  fs = {fs:.1f} Hz  |  duration = {times_min[-1]:.1f} min")

# ── Step 2: calibrate ──────────────────────────────────────────────────────────

window_size = int(WINDOW_SEC * fs)      # 200 at 20 Hz
step        = max(1, window_size // CALIBRATION_STEP_RATIO)
calib_needed = window_size + step * (CALIBRATION_MIN_WINDOWS - 1)
calib_data   = pressures[:calib_needed]
calib_end_min = times_min[calib_needed - 1]

print(f"  Calibrating on first {calib_needed} samples ({calib_end_min:.1f} min) ...")
detector = CloggingDetector(fs=fs, window_sec=WINDOW_SEC, sigma=SIGMA)
detector.calibrate(calib_data.tolist())

baseline_norm = detector.baseline_spectrum_norm.copy()   # B̂(f)
fft_threshold = detector.fft_threshold
freqs         = np.fft.rfftfreq(window_size, d=1.0 / fs)
print(f"  Threshold τ(σ=3) = {fft_threshold:.4f}")

# ── Step 3: full-pass — collect composite scores + spectrogram frames ──────────

# Spectrogram: compute one frame per `spec_step` samples to keep memory reasonable
spec_step         = max(1, int(fs * 5))   # one frame every 5 seconds
composite_times   = []
composite_scores  = []
spec_frames       = []      # list of P_norm arrays
spec_times_min    = []      # centre-time of each spectrogram frame

buf = collections.deque(maxlen=window_size)
snap_baseline  = baseline_norm.copy()  # will be overwritten with actual baseline
snap_detection = None                  # P_norm at first threshold crossing
snap_blockage  = None                  # P_norm nearest to confirmed blockage
detection_time_min = None

frame_counter = 0

for i, (t_min, p) in enumerate(zip(times_min, pressures)):
    buf.append(p)

    if len(buf) < window_size:
        continue

    sig = np.asarray(buf)
    _, _, P_norm = hamming_fft_norm(sig, fs)
    score = l1_distance(P_norm, baseline_norm)

    # Composite score time-series (every sample after calibration)
    if t_min >= calib_end_min:
        composite_times.append(t_min)
        composite_scores.append(score)

        # First threshold crossing
        if detection_time_min is None and score > fft_threshold:
            detection_time_min = t_min
            snap_detection     = P_norm.copy()
            print(f"  First crossing at {t_min:.2f} min  (score={score:.4f})")

    # Spectrogram frame
    if frame_counter % spec_step == 0:
        spec_frames.append(P_norm.copy())
        spec_times_min.append(t_min)

    frame_counter += 1

# Snapshot nearest to confirmed blockage
blockage_idx = np.argmin(np.abs(times_min - CONFIRMED_BLOCKAGE_MIN))
buf_block = collections.deque(pressures[max(0, blockage_idx - window_size):blockage_idx],
                               maxlen=window_size)
if len(buf_block) == window_size:
    _, _, snap_blockage = hamming_fft_norm(np.asarray(buf_block), fs)
    print(f"  Blockage snapshot at {times_min[blockage_idx]:.1f} min")

composite_times  = np.asarray(composite_times)
composite_scores = np.asarray(composite_scores)
spec_matrix      = np.asarray(spec_frames).T          # shape (freq_bins, time_frames)
spec_times_min   = np.asarray(spec_times_min)

# Apply 60-second rolling-median smoothing (matches Figure 5.5 in thesis)
# to find the sustained threshold crossing
smooth_win = int(60.0 * fs)   # 1200 samples at 20 Hz
from numpy.lib.stride_tricks import sliding_window_view
if len(composite_scores) > smooth_win:
    padded   = np.pad(composite_scores, (smooth_win // 2, smooth_win // 2), mode="edge")
    smoothed = np.median(sliding_window_view(padded, smooth_win), axis=-1)
    smoothed = smoothed[:len(composite_scores)]
else:
    smoothed = composite_scores.copy()

sustained_idx = np.argmax(smoothed > fft_threshold)
if smoothed[sustained_idx] > fft_threshold:
    sustained_crossing_min = float(composite_times[sustained_idx])
    # Use the sustained crossing time for the snapshot
    sustained_idx_global = np.searchsorted(times_min, sustained_crossing_min)
    buf_det = collections.deque(
        pressures[max(0, sustained_idx_global - window_size):sustained_idx_global],
        maxlen=window_size,
    )
    if len(buf_det) == window_size:
        _, _, snap_detection = hamming_fft_norm(np.asarray(buf_det), fs)
    print(f"  Sustained crossing (60 s median) at {sustained_crossing_min:.2f} min")
else:
    sustained_crossing_min = detection_time_min
    print("  No sustained crossing found, using first raw crossing")

# ── Figure 1: Spectrogram + Composite score ────────────────────────────────────

print("Generating spectrogram figure ...")

fig1, (ax_spec, ax_comp) = plt.subplots(
    2, 1, figsize=(11, 6),
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.35},
)

# — Spectrogram panel —
log_spec = np.log10(spec_matrix + 1e-12)
vmin, vmax = np.percentile(log_spec, [2, 98])
im = ax_spec.pcolormesh(
    spec_times_min, freqs, log_spec,
    shading="auto",
    cmap="viridis",
    vmin=vmin, vmax=vmax,
)
cbar = fig1.colorbar(im, ax=ax_spec, pad=0.01, fraction=0.025)
cbar.set_label("log$_{10}$ power", fontsize=9)

ax_spec.set_ylabel("Frequency (Hz)")
ax_spec.set_xlim(0, times_min[-1])
ax_spec.set_title("STFT Spectrogram — Experiment 2 ($f_s$ = 20 Hz)", fontsize=11)

# Annotation lines
for t, lbl, col in [
    (calib_end_min,              "Calibration\nend",      GRAY),
    (sustained_crossing_min,     "Composite\nalarm",       ORANGE),
    (CONFIRMED_BLOCKAGE_MIN,     "Confirmed\nblockage",    RED),
]:
    if t is not None:
        ax_spec.axvline(t, color=col, lw=1.4, ls="--", alpha=0.85)
        ax_spec.text(t + 1.0, freqs[-1] * 0.92, lbl, color=col,
                     fontsize=8, va="top", ha="left")

# — Composite score panel —
ax_comp.plot(composite_times, composite_scores, color=PURPLE, lw=0.6, alpha=0.4, label="Composite score $d_m$ (raw)")
ax_comp.plot(composite_times, smoothed, color=PURPLE, lw=1.4, label="60 s median")
ax_comp.axhline(fft_threshold, color=ORANGE, lw=1.3, ls="--",
                label=f"Threshold $\\tau_{{\\sigma=3}}$ = {fft_threshold:.3f}")
ax_comp.set_xlabel("Time (min)")
ax_comp.set_ylabel("$d_m$")
ax_comp.set_xlim(0, times_min[-1])
ax_comp.set_ylim(bottom=0)
ax_comp.legend(fontsize=9, loc="upper left")

for t, col in [
    (calib_end_min,              GRAY),
    (sustained_crossing_min,     ORANGE),
    (CONFIRMED_BLOCKAGE_MIN,     RED),
]:
    if t is not None:
        ax_comp.axvline(t, color=col, lw=1.2, ls="--", alpha=0.8)

fig1.savefig(OUT_SPECTROGRAM, bbox_inches="tight", dpi=150)
plt.close(fig1)
print(f"  Saved {OUT_SPECTROGRAM}")

# ── Figure 2: Three spectral snapshots ────────────────────────────────────────

print("Generating spectral snapshots figure ...")

fig2, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
fig2.subplots_adjust(wspace=0.35)

panels = [
    (snap_baseline,  "Baseline $\\hat{B}(f)$\n(calibration window mean)",  GREEN,  None),
    (snap_detection, f"Alarm snapshot\n($t$ = {sustained_crossing_min:.1f} min, sustained crossing)", ORANGE, snap_baseline),
    (snap_blockage,  f"Blockage snapshot\n($t$ ≈ {CONFIRMED_BLOCKAGE_MIN:.0f} min, confirmed blockage)", RED, snap_baseline),
]

for ax, (spectrum, title, color, baseline) in zip(axes, panels):
    if spectrum is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        continue

    ax.plot(freqs, spectrum, color=color, lw=1.5, label="$\\hat{P}(f)$")

    if baseline is not None:
        ax.plot(freqs, baseline, color=GREEN, lw=1.2, ls="--",
                alpha=0.7, label="$\\hat{B}(f)$ (baseline)")
        # Shade the L1 contribution per bin
        ax.fill_between(freqs, baseline, spectrum,
                         where=(spectrum > baseline),
                         color=color, alpha=0.25,
                         label=f"$|\\hat{{P}}-\\hat{{B}}|$ (+)")
        ax.fill_between(freqs, baseline, spectrum,
                         where=(spectrum < baseline),
                         color=BLUE, alpha=0.20,
                         label=f"$|\\hat{{P}}-\\hat{{B}}|$ (−)")
        dm = l1_distance(spectrum, baseline)
        ax.set_title(f"{title}\n$d_m$ = {dm:.3f}", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized power $\\hat{P}(f)$")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.legend(fontsize=8, loc="upper right")

fig2.suptitle(
    "Spectral shape evolution — Experiment 2 ($f_s$ = 20 Hz, $\\sigma$ = 3)\n"
    "Unit-sum normalized spectra at three key time points (Equations 2.4–2.7)",
    fontsize=10,
)

fig2.savefig(OUT_SNAPSHOTS, bbox_inches="tight", dpi=150)
plt.close(fig2)
print(f"  Saved {OUT_SNAPSHOTS}")
print("Done.")
