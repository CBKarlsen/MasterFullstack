# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This App Does

Real-time pipe clogging detection platform. Streams sensor data (pressure, flow) from CSV/Excel files, analyzes signals using FFT and spectral methods, runs pluggable ML models, and visualizes results in a web dashboard.

## Development Setup

Two servers must run concurrently:

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
2. **Calibration phase** (~400 samples / 20s of healthy baseline data) establishes sigma-based thresholds: `fft_threshold = baseline_mean + sigma * std`
3. **Real-time simulation**: WebSocket streams samples; backend computes static/composite/turbulence scores and spectral slope per sample, but sends only every 10th frame (~2Hz effective rate)
4. Active ML models receive `[static_score, composite_score, turbulence_score, spectral_slope]` and output clogging probability
5. Dashboard updates charts, traffic light (green/yellow/red), and ETA to failure
6. **Batch mode**: processes entire file at once with tunable sigma parameter

### Backend (`app/`)
| File | Role |
|------|------|
| `main.py` | FastAPI app, all REST endpoints and WebSocket handler |
| `engine.py` | `SimulationEngine` — orchestrates WebSocket streaming loop |
| `backend.py` | `CloggingDetector` — core signal processing and scoring logic |
| `dataloader.py` | `DataStreamer` — CSV/XLSX parsing with auto column detection |
| `batch.py` | Batch analysis processing |
| `models/registry.py` | `ModelRegistry` — dynamic model loading/hot-reload every 5s |
| `models/base.py` | `BaseModel` abstract class + `PredictionResult` + `ModelMetadata` |
| `models/builtin/fft_physics.py` | Built-in physics model (registered on startup) |
| `models/wrappers/` | Framework-specific adapters (sklearn, torch, keras) |
| `models/sequence_buffer.py` | Sliding window buffers for LSTM-style models |

### Frontend (`frontend/src/`)
| File | Role |
|------|------|
| `store/simulationStore.ts` | Zustand global state; rolling buffer capped at 5000 chart points |
| `hooks/useWebSocket.ts` | WebSocket + message batching via `requestAnimationFrame` (60 FPS flush) |
| `components/Dashboard.tsx` | Main orchestrator component |
| `components/ControlChart.tsx` | QA/QC control charts |
| `components/Batchanalysis.tsx` | Batch processing UI |

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

### Non-obvious Behaviors

**Scoring methods in `backend.py` (`CloggingDetector`):**
- **Static**: `abs(current_window_mean - baseline_mean)`
- **Composite**: `high_energy / low_energy` FFT ratio, split at 25% of Nyquist. Ratio > 0.05 signals anomaly
- **Turbulence**: Detrended FFT with Hamming window applied
- **Spectral slope**: Log-log linear fit on 1–50 Hz band. Healthy pipe: −2.5 to −3.0 (Kolmogorov cascade); clogged: −1.0 to −1.5 (white noise)
- **ETA**: Log-linear regression on last 150 composite values; unreliable when trend slope ≤ 0

**Frontend message batching (`useWebSocket.ts`):** Incoming WebSocket messages are buffered and flushed once per `requestAnimationFrame` (~60Hz). React state only updates with the latest buffered message, preventing re-render thrashing at 20Hz wire rate.

**`DataStreamer` auto-detection:** Sampling rate inferred from median of first 100 time-column deltas. Handles European decimals (comma → dot). Extracts ALL numeric columns into a `raw` dict that passes through the entire pipeline to the frontend, enabling visualization of arbitrary sensor channels beyond flow/pressure.

**Sigma recalculation:** Changing sigma mid-stream immediately recalculates all thresholds from stored baseline statistics — no re-calibration or re-processing needed.

**Registry singleton:** `get_registry(models_dir)` returns a single instance. The built-in FFT physics model is registered at startup. Hot-reload background thread polls every 5s.

**Sequence model buffering:** Models show "filling (N%)" status until their sliding window buffer is full. `MultiModelSequenceManager` maintains a separate buffer per sequence model.
