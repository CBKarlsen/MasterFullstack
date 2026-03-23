# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This App Does

Real-time pipe clogging detection platform for multiphase flow in oil pipelines. Streams sensor data (pressure, flow) from CSV/Excel files, analyzes signals using five complementary detection methods, runs pluggable ML models, and visualizes results in a web dashboard. Built as a master's thesis artifact (Casper Benjamin Karlsen, HVL) using Design Science Research methodology.

## Development Setup

**Always use a virtual environment.** PyWavelets and other packages must be installed inside the venv — not system Python — or `_PYWT_AVAILABLE` will be False and the wavelet method will silently produce zeros.

**Backend (FastAPI, port 8000):**
```bash
pip install -r requirements.txt
python -m app.main
```

**Frontend (React + Vite, port 3000):**
```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api/*` HTTP requests to `localhost:8000`. WebSocket connects directly to `ws://localhost:8000/ws/simulate` (dev bypasses Vite proxy).

**Lint frontend:**
```bash
cd frontend && npm run lint
```

## Architecture

### Data Flow
1. User selects a CSV/XLSX file with columns: time, flow, p_in, p_out
2. **Calibration phase** (~400 samples / 20s of healthy baseline data) builds:
   - Baseline spectral fingerprint (FFT) for composite method
   - Baseline wavelet detail-energy distribution (DWT) for wavelet method
   - Thresholds via `fft_threshold = percentile(baseline_composites, sigma_to_pct(sigma))` — percentile-based, not Gaussian
3. **Real-time simulation**: WebSocket streams samples; backend computes all 5 scores per sample, sends every 10th frame (~2Hz effective rate)
4. Active ML models receive `[static_score, composite_score, turbulence_score, spectral_slope, wavelet_score]` and output clogging probability
5. Dashboard updates charts, traffic light (green/yellow/red), and ETA to failure
6. **Batch mode**: processes entire file at once with tunable sigma parameter; shows all 5 method charts side-by-side for comparison

### Backend (`app/`)
| File | Role |
|------|------|
| `main.py` | FastAPI app, all REST endpoints and WebSocket handler |
| `engine.py` | `SimulationEngine` — orchestrates WebSocket streaming loop |
| `backend.py` | `CloggingDetector` — core signal processing and all 5 scoring methods |
| `dataloader.py` | `DataStreamer` — CSV/XLSX parsing with auto column detection |
| `batch.py` | Batch analysis processing |
| `models/registry.py` | `ModelRegistry` — dynamic model loading/hot-reload every 5s |
| `models/base.py` | `BaseModel` abstract class + `PredictionResult` + `ModelMetadata` |
| `models/builtin/random_forest.py` | Built-in RF model, retrainable via UI |
| `models/builtin/isolation_forest.py` | Built-in unsupervised anomaly detector, needs training on healthy data |
| `models/wrappers/` | Framework-specific adapters (sklearn, torch, keras) |
| `models/sequence_buffer.py` | Sliding window buffers for LSTM-style models |

### Frontend (`frontend/src/`)
| File | Role |
|------|------|
| `store/simulationStore.ts` | Zustand global state; rolling buffer capped at 5000 chart points |
| `hooks/useWebSocket.ts` | WebSocket + message batching via `requestAnimationFrame` (60 FPS flush) |
| `components/Dashboard.tsx` | Main orchestrator component |
| `components/ControlChart.tsx` | QA/QC control charts (used by both live and batch views) |
| `components/Batchanalysis.tsx` | Batch processing UI — all 5 method charts + threshold crossing cards |
| `components/ModelsTab.tsx` | ML model management UI |

**Frontend stack:** React 19, TypeScript, Vite, Zustand, TanStack Query, Axios, Recharts.

### Key API Endpoints
- `GET /api/data` — list available data files
- `POST /api/data/upload` — upload CSV/XLSX
- `DELETE /api/data/{filename}` — remove data files
- `GET /api/data/{filename}/columns` — get column metadata
- `WS /ws/simulate` — real-time streaming simulation
- `POST /api/analyze` — run batch analysis
- `POST /api/analyze/thresholds` — recalculate thresholds with new sigma (no reprocessing)
- `GET /api/models` — list models with stats (prediction count, avg inference time)
- `PUT /api/models/{name}/enable` — toggle model on/off
- `POST /api/models/upload` — upload model file
- `DELETE /api/models/{name}` — unregister model (does not delete file)

### WebSocket Protocol
Client sends: `{"action": "start", "file": "filename.csv", "speed": 1.0}` (speed clamps to 0.1–100x).
Server sends JSON frames with type field: `"data"` | `"status"` | `"columns"`.

### ML Model System

Drop model files into `models/` — the registry hot-reloads them within 5 seconds. Supported formats:
- `.pkl` — scikit-learn (via joblib)
- `.pt`/`.pth` — PyTorch (full model save, not state_dict)
- `.h5` — TensorFlow/Keras

Optionally add a `.json` metadata file with the same name for custom configuration. Set `"input_type": "sequence"` and `"sequence_length": N` for LSTM-style models that receive `(N, 4)` arrays instead of `(4,)` vectors. Without a metadata file, defaults are auto-generated.

To implement a custom model, subclass `BaseModel` from `app/models/base.py`. Required: `predict(features)` returning `PredictionResult`. Optional: `warmup()` and `cleanup()`.

---

## The Five Detection Methods

All methods are computed in `CloggingDetector.process_sample()` in `backend.py`. All five scores are included in the `_features` dict passed to ML models.

### 1. Static (Hydraulic Head Deviation)
`abs(current_window_mean - baseline_mean)`

Detects sustained DC shift in pressure drop. Simple but slow — only triggers on significant mean-level changes (e.g. full blockage).

### 2. Composite (L1 Spectral Distance)
L1 distance between the normalised current FFT power spectrum and the normalised baseline spectrum fingerprint.

- Score ≈ 0 for healthy; rises as spectral *shape* deviates from baseline
- Calibrated per-file so files with different spectral characters don't cross-contaminate
- Baseline adapts via EMA (default τ=5 min, alpha=1/(fs×60×5)) to track slow drift; **only adapts when score < 40% of threshold** to prevent clogging events from corrupting the reference
- Threshold: `np.percentile(_sorted_composites, sigma_to_pct(sigma))` — distribution-free
- `_sorted_composites` is pre-sorted at calibration time for O(1) percentile lookup when sigma changes

### 3. Turbulence (Detrended FFT High-Band Energy)
DC-removed signal, Hamming-windowed FFT; score = fraction of energy above the low-frequency split.

Measures short-scale turbulence intensity. Rises when vortex shedding and cavitation increase near a partial blockage.

### 4. Spectral Slope (Log-Log Fit)
Linear fit of `log(power)` vs `log(freq)` over 1 Hz–Nyquist.

- Healthy Kolmogorov turbulent cascade: slope ≈ −2.5 to −3.0
- Clogged / white-noise dominated: slope ≈ −1.0 to −1.5
- Not threshold-based; used as a feature for ML models

### 5. Wavelet (DWT Detail-Energy Distribution Shift)
`CloggingDetector.calculate_wavelet_score(signal)` — requires PyWavelets (`pywt`).

Uses `pywt.wavedec(signal, 'db4', level=4)` producing `[cA4, cD4, cD3, cD2, cD1]`.

**Key design choice:** the approximation coefficient cA4 is **excluded** from the score. It captures slow bulk-flow variation that is stable regardless of clogging. Only the four detail levels `[cD4, cD3, cD2, cD1]` (low→high frequency turbulent scales) are used.

Score = L1 distance between EMA-smoothed current detail-energy distribution and baseline:
- Score ≈ 0 → detail energy cascade matches baseline (healthy)
- Score → 2 → distribution fully inverted (maximum anomaly)

**EMA smoothing (`_wavelet_ema_alpha = 0.05`):** cD4 has only ~12 DWT coefficients per 200-sample window — too few for a stable per-sample energy estimate. An EMA with α=0.05 smooths over ~20 samples (1s at 20Hz), suppressing noise while tracking sustained clogging shifts. Both calibration (EMA replayed over windows) and analysis use the same α so thresholds remain calibrated.

Clogging shifts energy toward cD1/cD2 (high-frequency turbulence at the blockage site). This is complementary to Composite (which measures FFT shape in the frequency domain) — Wavelet measures energy redistribution across *scales* in the time-frequency domain.

Threshold uses the same percentile approach as Composite, computed from `_sorted_wavelet_composites` (built from EMA-smoothed calibration scores).

**If pywt is not installed in the active venv**, `_PYWT_AVAILABLE = False` and `calculate_wavelet_score` silently returns 0.0. Fix: `pip install PyWavelets` inside the venv.

---

## Non-obvious Behaviors

**Threshold recalculation:** Changing sigma recalculates all three thresholds (static, composite, wavelet) using `np.percentile` on pre-sorted arrays — O(1), no re-calibration needed.

**Calibration window density:** Step = `window_size // 20` (5% overlap), giving ~21 windows per calibration period. This produces a robust baseline distribution for both FFT and wavelet methods.

**Batch ML inference:** `_apply_batch_ml()` in `batch.py` calls `predict_batch()` once per model over the full feature matrix — vectorised, orders of magnitude faster than per-sample inference.

**Frontend sigma recalculation (batch):** When sigma changes in the batch UI, thresholds are recomputed client-side using `mean + sigma * std` (approximate). The accurate percentile threshold requires re-running the full analysis.

**Traffic light** turns non-green at 70% of `fft_threshold` (not 20%).

**ETA prediction:** Log-linear regression on the last 150 composite buffer values. Unreliable when trend slope ≤ 0; returns gray "Stable (No Growth)".

**Frontend message batching (`useWebSocket.ts`):** Messages buffered and flushed once per `requestAnimationFrame` (~60Hz). React state only updates with the latest buffered message, preventing re-render thrashing at 20Hz wire rate.

**`DataStreamer` auto-detection:** Sampling rate inferred from median of first 100 time-column deltas. Handles European decimals (comma → dot). All numeric columns pass through to frontend as `raw` dict.

**Registry singleton:** `get_registry()` returns a single instance. Built-in models registered at startup. Hot-reload polls every 5s.

**Sequence model buffering:** `MultiModelSequenceManager` maintains a separate sliding window buffer per sequence model. Shows "filling (N%)" until ready.

**Isolation Forest** is dormant until trained on healthy data via the Models tab. Returns 0 probability until trained.

---

## Thesis Context

This platform is Casper's master's thesis software artifact. Key confirmed finding: on a ~4h21m file (1.5M points), the composite method detects a **partial clogging event at ~1h06m** — evidenced by simultaneous step-down in flow rate (~5%) and permanent elevation in composite score. The static method did not trigger, confirming the composite method's superiority for early-stage partial blockage detection. Video footage review pending to confirm physical blockage at that timestamp.

The five methods are designed for comparative evaluation in the thesis: which detects earliest, which has fewest false positives, and how they complement each other.
