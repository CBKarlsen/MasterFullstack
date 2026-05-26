"""
Performance benchmarks for the clogging-detection platform (thesis NFR evidence).

Run from the repository root with the project venv:

    ./venv/bin/python benchmarks_nfr.py

Produces:
  A. Per-sample processing time / throughput across the five corpus sampling
     rates  -> NFR-1 (throughput) and the compute share of NFR-2 (latency).
  B. No-lost-samples integrity check                              -> NFR-1.
  C. Batch vs streaming-style replay equivalence + determinism    -> NFR-8.

All measurements are *artificial* in FEDS terms: they characterise the
implementation under controlled replay on this machine, not a deployed plant.
"""

import time
import platform
import numpy as np

from app.backend import CloggingDetector
from app.batch import run_batch_analysis
from app.engine import detect_sampling_rate
from app.dataloader import DataStreamer

CORPUS_RATES = [0.4, 1.0, 1.11, 20.0, 100.0]   # the five distinct fs in the corpus
DATA_FILE = "data/0709.csv"                      # tracked 1 Hz confirmed-blockage file


def synth_healthy(n, fs, seed=0):
    """Synthetic healthy DeltaP: stable mean + slow pump tone + sensor noise.
    Timing depends on array sizes, not signal content, so this is fine for
    measuring per-sample cost."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    sig = 5.0 + 0.3 * np.sin(2.0 * np.pi * 0.1 * t) + 0.05 * rng.standard_normal(n)
    return sig, t


def bench_per_sample(fs, timed=3000):
    det = CloggingDetector(fs=fs, window_sec=30.0, sigma=3.0, enable_models=False)
    cal_n = CloggingDetector.required_calibration_samples(fs)
    cal_vals, _ = synth_healthy(cal_n, fs, seed=1)
    det.calibrate(list(cal_vals))

    W = det.window_size
    vals, times = synth_healthy(W + timed + 5, fs, seed=2)

    # Warm up: fill the sliding-window buffer (process_sample returns None until full)
    for i in range(W):
        det.process_sample(float(vals[i]), float(times[i]))

    durs = np.empty(timed)
    for k, i in enumerate(range(W, W + timed)):
        s = time.perf_counter()
        det.process_sample(float(vals[i]), float(times[i]))
        durs[k] = (time.perf_counter() - s) * 1000.0  # ms

    mean = float(np.mean(durs))
    budget = 1000.0 / fs  # ms available per sample in real time
    return {
        "fs": fs,
        "window": W,
        "skip": det._analysis_skip,
        "mean": mean,
        "median": float(np.median(durs)),
        "p95": float(np.percentile(durs, 95)),
        "p99": float(np.percentile(durs, 99)),
        "max": float(np.max(durs)),
        "throughput": 1000.0 / mean,         # samples/s the backend can sustain
        "budget": budget,
        "headroom": budget / mean,           # x real-time
    }


def test_no_lost_samples(fs=20.0, n=6000):
    det = CloggingDetector(fs=fs, window_sec=30.0, sigma=3.0, enable_models=False)
    cal_n = CloggingDetector.required_calibration_samples(fs)
    cal_vals, _ = synth_healthy(cal_n, fs, seed=1)
    det.calibrate(list(cal_vals))
    W = det.window_size
    vals, times = synth_healthy(n, fs, seed=3)
    results = [det.process_sample(float(vals[i]), float(times[i])) for i in range(n)]
    produced = sum(1 for r in results if r is not None)
    expected = n - (W - 1)   # one result per sample once the buffer is full
    return produced, expected, (produced == expected)


def _batch_composite(path, sigma=3.0):
    res = run_batch_analysis(path, sigma=sigma)
    return [p["composite_score"] for p in res["timeseries"] if p["phase"] == "analysis"]


def _streaming_replay_composite(path, sigma=3.0):
    """Mirror the engine's calibration sizing, then drive a plain process_sample
    loop (the streaming computation, minus the websocket/asyncio wrapper)."""
    pts = list(DataStreamer(path).stream())
    dP = [float(p.get("p_in", 0.0)) - float(p.get("p_out", 0.0)) for p in pts]
    tt = [float(p.get("time", 0.0)) for p in pts]

    # Replicate batch/engine phase-1 calibration sizing exactly.
    buf_t = []
    cal_count = len(dP)
    for i in range(len(pts)):
        buf_t.append(tt[i])
        if len(buf_t) >= 100:
            needed = CloggingDetector.required_calibration_samples(detect_sampling_rate(buf_t))
            if len(buf_t) >= needed:
                cal_count = len(buf_t)
                break
    fs = detect_sampling_rate(buf_t)

    det = CloggingDetector(fs=fs, window_sec=30.0, sigma=sigma, enable_models=False)
    det.calibrate(dP[:cal_count])
    comp = []
    for i in range(cal_count, len(dP)):
        r = det.process_sample(dP[i], tt[i])
        if r is not None:
            comp.append(r["composite"])
    return comp, fs, cal_count


def main():
    print("=" * 70)
    print("CLOGGING-DETECTION PLATFORM — PERFORMANCE BENCHMARKS")
    print(f"machine: {platform.platform()}")
    print(f"python : {platform.python_version()} | numpy {np.__version__}")
    print("=" * 70)

    # ---- A. per-sample latency / throughput ----------------------------------
    print("\n[A] Per-sample processing time and sustained throughput\n")
    hdr = f"{'fs (Hz)':>8} {'window':>7} {'skip':>5} {'mean(ms)':>9} {'p95(ms)':>8} {'p99(ms)':>8} {'thru(s^-1)':>11} {'budget(ms)':>11} {'headroom':>9}"
    print(hdr)
    print("-" * len(hdr))
    rowsA = []
    for fs in CORPUS_RATES:
        r = bench_per_sample(fs)
        rowsA.append(r)
        print(f"{r['fs']:>8} {r['window']:>7} {r['skip']:>5} {r['mean']:>9.4f} "
              f"{r['p95']:>8.4f} {r['p99']:>8.4f} {r['throughput']:>11.0f} "
              f"{r['budget']:>11.1f} {r['headroom']:>8.0f}x")

    # ---- B. no-lost-samples --------------------------------------------------
    print("\n[B] No-lost-samples integrity (NFR-1)\n")
    produced, expected, ok = test_no_lost_samples()
    print(f"    produced results = {produced}, expected = {expected} -> "
          f"{'PASS' if ok else 'FAIL'}")

    # ---- C. batch vs streaming + determinism (NFR-8) -------------------------
    print("\n[C] Batch vs streaming equivalence and determinism (NFR-8)\n")
    try:
        b1 = _batch_composite(DATA_FILE)
        b2 = _batch_composite(DATA_FILE)
        det_ok = len(b1) == len(b2) and float(np.max(np.abs(np.array(b1) - np.array(b2)))) == 0.0
        print(f"    determinism: batch run twice, n={len(b1)}, "
              f"max|diff|={np.max(np.abs(np.array(b1)-np.array(b2))) if len(b1)==len(b2) else 'len mismatch'} -> "
              f"{'PASS' if det_ok else 'FAIL'}")

        sc, fs_s, cal = _streaming_replay_composite(DATA_FILE)
        n = min(len(b1), len(sc))
        if n == 0:
            print("    equivalence: no analysis points produced -> SKIP")
        else:
            diff = float(np.max(np.abs(np.array(b1[:n]) - np.array(sc[:n]))))
            eq_ok = (len(b1) == len(sc)) and diff < 1e-9
            print(f"    equivalence: batch n={len(b1)} vs streaming-replay n={len(sc)} "
                  f"(fs={fs_s:.3f}, calib={cal}), max|diff|={diff:.3e} -> "
                  f"{'PASS' if eq_ok else 'FAIL'}")
    except Exception as exc:
        print(f"    [C] could not run on {DATA_FILE}: {exc}")

    # ---- LaTeX table for part A (paste-ready) --------------------------------
    print("\n[LaTeX] Part-A table body (paste into the thesis):\n")
    for r in rowsA:
        fs_str = f"{r['fs']:g}"
        print(f"{fs_str} & {r['window']} & {r['mean']:.3f} & {r['p99']:.3f} & "
              f"{r['throughput']:.0f} & {r['budget']:.1f} & {r['headroom']:.0f}$\\times$ \\\\")
    print()


if __name__ == "__main__":
    main()
