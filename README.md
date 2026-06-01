# Pipe Clogging Detection Platform

A real-time monitoring and analysis platform for detecting and predicting clogging in oil pipelines. Built as the software artifact for a master's thesis at HVL (Western Norway University of Applied Sciences) by Casper Benjamin Karlsen.

The core idea: instead of waiting for the average pressure to drop, the platform listens to the *texture* of the flow — the frequency signature of the pressure signal — and notices when that signature starts to shift. In the thesis experiments this gave warning of a partial blockage roughly an hour before a traditional pressure alarm would have triggered.

---

## Contents

- [What problem does this solve?](#what-problem-does-this-solve)
- [The big picture](#the-big-picture)
- [What is FFT and why does it matter?](#what-is-fft-and-why-does-it-matter)
- [The four detection methods](#the-four-detection-methods)
- [The sigma parameter](#the-sigma-parameter)
- [Clogging trajectory forecast](#clogging-trajectory-forecast)
- [Machine learning integration](#machine-learning-integration)
- [Getting started](#getting-started)
- [Input data format](#input-data-format)
- [API reference](#api-reference)
- [Repository layout](#repository-layout)
- [Reproducing the thesis figures](#reproducing-the-thesis-figures)
- [Tech stack](#tech-stack)
- [Thesis context](#thesis-context)

---

## What problem does this solve?

In oil and gas pipelines, partial blockages — sand deposits, wax build-up, hydrate formation — can go undetected for hours before causing a full shutdown. By the time a simple pressure sensor triggers an alarm, the blockage is often already severe.

This platform detects clogging much earlier, by analysing how the flow *sounds* rather than just how hard it is pushing. It can tell that something inside the pipe is changing long before a traditional pressure threshold would fire.

---

## The big picture

```
Sensor data (CSV/Excel)
        │
        ▼
┌───────────────────┐
│  Calibration      │  First 20–30 seconds: learn what
│  (healthy flow)   │  "normal" looks like for this pipe
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Analysis         │  Every new data point is compared
│  (live scoring)   │  against the healthy baseline
└─────────┬─────────┘
          │
          ▼
┌───────────────────────────────────────┐
│  4 detection methods (run in parallel) │
│  • Static       • Composite (FFT)      │
│  • Turbulence   • Spectral Slope       │
└─────────┬─────────────────────────────┘
          │
          ▼
┌───────────────────┐
│  ML ensemble      │  Optional: trained models vote on
│  (optional)       │  whether clogging is occurring
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Dashboard        │  Live charts, traffic light,
│  & forecast       │  and time-to-critical prediction
└───────────────────┘
```

The platform runs in two modes that share the exact same detection code, so their scores are directly comparable:

- **Live mode** streams sensor data in real time over a WebSocket and updates the dashboard continuously. The backend analyses up to ~20 samples per second and pushes batched frames to the browser at roughly 2 Hz so the UI stays smooth.
- **Batch mode** processes an entire file in one pass and returns the full result set. This is what you use for post-hoc analysis, parameter tuning, sigma sweeps, and thesis evaluation.

---

## What is FFT and why does it matter?

### Sound as an analogy

Imagine listening to a musical chord through a wall. You can't see what is happening on the other side, but you can hear *how the sound changes*. If the instruments suddenly sound different — a new buzzing frequency appears, or a familiar tone goes quiet — you know something has changed even though you can't see it.

Pipelines are similar. When fluid flows through a healthy pipe, it produces a characteristic pattern of vibrations and pressure fluctuations. This pattern has a "signature": some frequencies are strong, others weak. When a blockage starts forming, the flow is disrupted, turbulence changes, new vortices appear around the obstruction, and the pressure-fluctuation signature shifts.

**FFT (Fast Fourier Transform)** is the mathematical tool that reads this signature.

### What FFT actually does

A pressure sensor records one number every 0.05 seconds (at 20 Hz): `[0.412, 0.415, 0.409, 0.421, ...]`. This is the **time-domain signal**: it shows how pressure changes over time, but it is hard to read directly because it is just a noisy wiggly line.

FFT converts that wiggly line into a **frequency breakdown** — it answers the question *"how much of this signal is slow oscillation, and how much is fast oscillation?"* Think of a prism splitting white light into a rainbow: the time-domain signal is the white light, and FFT splits it into its component frequencies.

```
Time domain:                    Frequency domain (after FFT):
                                     │
   │  /\/\/\/\  /\/\               ██│
   │ /         /    \              ██│ ██
   │/               \              ██│ ████
   └───────────────── time         └─────────── frequency
   (noisy, hard to read)           (clear peaks, easy to compare)
```

### How this platform uses FFT

During **calibration**, the platform records the FFT spectrum of healthy flow — the "fingerprint" of a normal, unclogged pipe — and stores it as a reference.

During **analysis**, it continuously computes the FFT spectrum of the current signal and asks: *how different is this from the healthy fingerprint?* The difference is measured as an **L1 distance**, essentially the total area between the two normalized spectrum shapes. A small distance means the flow looks healthy; a large, sustained distance means the spectral character of the flow has changed.

This is what the platform calls the **composite score**, and it is the most sensitive detection method available.

### Why this beats watching pressure alone

A simple pressure alarm fires only when the *average* pressure drop changes by a large amount, which happens only once a blockage is severe. The composite method detects changes in the *shape* of the pressure-fluctuation spectrum, which shifts much earlier — sometimes hours before a pressure alarm would trigger.

**Confirmed result on a real 4h21m dataset:** the composite method detected a partial blockage at ~1h06m. The traditional pressure-deviation method never triggered at all during the entire recording.

---

## The four detection methods

Each method looks at a different physical aspect of the flow. They are designed to complement each other — no single method is perfect, but together they provide strong evidence. All four are implemented in [`app/backend.py`](app/backend.py).

> **Evaluation scope:** Only the **static** and **composite** methods were formally tested and evaluated in the thesis. These are the two methods that batch mode treats as detectors — it computes their threshold crossings and drives the forecast from them, and they are the methods compared in the experiments below. The **turbulence** and **spectral slope** scores are computed and displayed, but they were used only as supporting signals and ML input features; they were **not** evaluated as standalone detectors.

### 1. Static — hydraulic head deviation

**What it measures:** the change in average pressure drop compared to the healthy baseline.

**Physics:** a blockage physically restricts the pipe. As the restriction grows, more pressure is needed to push the same flow through (Hagen–Poiseuille). The static score is how far the current mean pressure has drifted from baseline.

- **Strength:** highly interpretable, directly physical.
- **Weakness:** only fires on significant, sustained mean-level changes. A partial blockage that changes flow texture without shifting the average pressure will not trigger it.
- **Evaluation:** tested and evaluated in the thesis as the traditional pressure-based baseline.

### 2. Composite — L1 spectral distance (FFT-based)

**What it measures:** how much the shape of the FFT power spectrum has changed from the healthy baseline.

**Physics:** turbulence in a healthy pipe follows a predictable energy cascade (Kolmogorov: energy flows from large slow eddies to small fast ones). A partial blockage disrupts this cascade — the obstruction sheds new vortex frequencies, pushes energy toward higher frequencies, and alters the spectral texture.

**How it works:**
1. During calibration, the platform builds an average "healthy spectrum fingerprint" from many windows of healthy flow.
2. During analysis, every new window is FFT-transformed and compared to that fingerprint.
3. The score is the L1 distance between the current normalized spectrum and the fingerprint.
4. If the score stays above the threshold for a sustained period, an alert fires.

- **Strength:** sensitive to early-stage partial blockages; detects changes in flow texture long before the average pressure moves.
- **Weakness:** needs a clean calibration period of healthy flow.
- **Evaluation:** tested and evaluated in the thesis; this is the platform's primary detection result.

> Note: the FFT keeps its DC component (the signal is not detrended) because the rig runs a constant-Hz pump, which makes the mean pressure level itself part of the clogging signature. The turbulence method below detrends separately.

### 3. Turbulence — detrended FFT high-band energy

**What it measures:** how much of the signal's energy sits in the high-frequency band (fast oscillations) versus the low-frequency band.

**Physics:** a partial blockage acts like a rough obstacle. Fluid forced around it breaks into small, fast vortices, raising high-frequency turbulent energy. This method removes the low-frequency trend and measures the fraction of remaining energy at high frequencies.

- **Strength:** directly reflects turbulence intensity near a blockage.
- **Weakness:** can react to flow-rate changes that are unrelated to clogging.
- **Evaluation:** not assessed as a standalone detector in the thesis (supporting signal / ML feature only).

### 4. Spectral slope — power-law fit

**What it measures:** the steepness of the frequency spectrum on a log-log scale.

**Physics:** healthy turbulent flow produces a power spectrum that falls off at a characteristic rate — a straight line of slope roughly −2.5 to −3.0 on a log-log plot. A clogged pipe with disrupted turbulence has a flatter spectrum (closer to −1.0 to −1.5) because energy is spread more evenly across frequencies.

- **Strength:** a physically grounded single number that characterises the turbulence regime.
- **Use:** an input feature for the ML models rather than a standalone threshold detector.
- **Evaluation:** not assessed as a standalone detector in the thesis (supporting signal / ML feature only).

---

## The sigma parameter

Every threshold in the platform is expressed in units of **sigma (σ)** — how many standard deviations above the calibrated healthy baseline a score must rise before it counts as a detection. The static threshold is `σ × baseline_std`; the composite threshold is set at the matching one-tailed normal percentile of the calibration distribution (so the same knob stays intuitive for both). You can change σ live from the dashboard or per-run in batch mode.

Sigma trades sensitivity against false alarms:

- **Low σ (e.g. 2–3):** very sensitive, detects earlier, but more prone to premature/false alarms.
- **High σ (e.g. 5):** conservative, fewer false alarms, but may detect later or not fit a forecast.

The `sigma_*` scripts in the repository sweep and visualise this trade-off across the experiments (see [Reproducing the thesis figures](#reproducing-the-thesis-figures)).

---

## Clogging trajectory forecast

Once a method crosses its threshold, the platform fits three growth models to the composite-score trajectory after onset and projects when the score will reach a critical level (default: 2× the detection threshold). See [`app/forecast.py`](app/forecast.py).

| Model | Equation | Suitable when |
|-------|----------|---------------|
| Linear | `score = a·t + b` | Blockage grows at a constant rate (steady deposition) |
| Exponential | `score = s₀·eᵏᵗ` | Blockage accelerates over time (autocatalytic growth) |
| Power law | `score = a·tⁿ` | Aggregation kinetics (particle clustering) |

Each model reports its R² (goodness of fit) and an estimated time to the critical threshold. The consensus ETA is the median across the models that project a future crossing. Fits are weighted toward recent data and smoothed with a causal rolling median before fitting.

**If no model projects a crossing,** the result "No critical crossing projected" means the partial blockage has stabilised — the pipe is restricted but the restriction is not growing. That is itself a scientifically meaningful finding.

---

## Machine learning integration

Three models ship with the platform, managed by a registry in [`app/models/`](app/models/). They take the four detection scores as input features and output a clogging probability.

- **FFT-physics model** — a built-in rule-based model that maps the physics scores to a probability without any training.
- **Random Forest** (supervised) — trained on labelled data (healthy vs. clogged). Train it from the **Models** tab by selecting a data file; the backend labels samples from the traffic-light state and trains in a background thread while the UI polls for progress.
- **Isolation Forest** (unsupervised) — trained only on healthy data, no labels needed. It flags samples that look anomalous compared to normal flow.

All active models vote in a weighted ensemble. Each model's contribution is weighted by its own confidence multiplied by a user-set **trust weight** (0–5) from the Models tab, so you can amplify or mute any model without retraining or disabling it.

**Custom models** can be added by dropping files into the `models/` directory:

| File type | Framework | Wrapper |
|-----------|-----------|---------|
| `.pkl` | scikit-learn | `sklearn_wrapper.py` |
| `.pt` / `.pth` | PyTorch | `pytorch_wrapper.py` |
| `.h5` | TensorFlow / Keras | `tensorflow_wrapper.py` |

The registry watches the folder and hot-reloads new models within ~5 seconds. Optional metadata (name, author, description, input features) can be supplied via a sidecar `.json` file or the upload form.

---

## Getting started

### Requirements

- Python 3.10+ with a virtual environment
- Node.js 18+

### Backend

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend (port 8000)
python -m app.main
```

### Frontend

```bash
cd frontend
npm install
npm run dev                        # Starts on port 3000
```

Open `http://localhost:3000`. The Vite dev server proxies API and WebSocket calls to the backend on port 8000, so you only need to open the frontend URL.

---

## Input data format

CSV or Excel files with the following columns (auto-detected, case-insensitive, European decimal commas handled). Place files in the `data/` directory; a folder of files is streamed as one continuous recording.

| Column | Description | Example names |
|--------|-------------|---------------|
| Time | Timestamp in seconds | `time`, `t`, `Tid`, `Date/Time` |
| Flow rate | Mass or volumetric flow | `Flow rate`, `Flow rate (Mean)` |
| Inlet pressure | Pressure at pipe entry | `TS inlet pressure`, `Pressure after pump` |
| Outlet pressure | Pressure at pipe exit | `TS outlet pressure`, `Pressure before pump` |

The sampling rate is inferred from the timestamp column (falling back to 20 Hz). Any additional numeric columns are passed through to the dashboard as raw data for plotting.

---

## API reference

The backend is a FastAPI app ([`app/main.py`](app/main.py)). Interactive docs are available at `http://localhost:8000/docs` when it is running.

**Health**
- `GET /` — basic status
- `GET /health` — model counts

**Live streaming**
- `WS /ws/simulate` — connect, then send `{"action": "start", "file": "<name>", "speed": 1.0}`; the server streams result frames until the file ends or you send `{"action": "stop"}`.

**Batch analysis**
- `POST /api/analyze?file=<name>&sigma=3` — run a full-file analysis and return the complete time series
- `POST /api/analyze/thresholds` — recompute thresholds for given baseline stats and sigma

**Data files**
- `GET /api/data` — list files/folders in `data/`
- `POST /api/data/upload` — upload a CSV/Excel file
- `GET /api/data/{filename}/columns` — column metadata
- `DELETE /api/data/{filename}` — delete a file

**Models**
- `GET /api/models`, `GET /api/models/{name}`, `GET /api/models/{name}/stats`
- `PUT /api/models/{name}/enable`, `PUT /api/models/{name}/weight`
- `POST /api/models/upload`, `DELETE /api/models/{name}`
- `POST /api/models/random_forest/train`, `.../isolation_forest/train` (background jobs)
- `GET /api/models/{rf|if}/training-progress`, `POST /api/models/{rf|if}/reset`

---

## Repository layout

```
├── app/                            # FastAPI backend (the main artifact)
│   ├── main.py                     # API endpoints + WebSocket
│   ├── engine.py                   # Real-time streaming simulation loop
│   ├── backend.py                  # CloggingDetector: calibration + 4 methods
│   ├── batch.py                    # Whole-file (offline) analysis
│   ├── forecast.py                 # Growth-model ETA forecasting
│   ├── dataloader.py               # CSV/Excel parsing + auto-detection
│   ├── ml_interface.py             # Legacy standalone predictor (unused)
│   └── models/                     # ML registry, wrappers, built-in models
├── frontend/                       # React + TypeScript + Vite dashboard
│   └── src/
│       ├── components/             # UI: charts, panels, model management
│       ├── store/                  # Zustand global state
│       ├── hooks/                  # WebSocket hook
│       └── utils/                  # Forecast client, CSV import/export
├── data/                           # Sensor recordings (CSV/Excel)
├── models/                         # Drop custom ML model files here (hot-reloaded)
├── benchmarks_nfr.py               # Non-functional performance benchmark
├── sigma_*.py                      # Sigma sweep / analysis / figure scripts
└── *.py                            # Standalone thesis figure generators
```

> `app/ml_interface.py` is an early prototype that predates the model registry and is no longer imported anywhere. It is kept for reference only and can be deleted.

---

## Reproducing the thesis figures

The standalone scripts in the repository root regenerate the figures and tables used in the thesis. Each has a docstring describing its inputs and outputs. Run them from the project root with the virtual environment active (`python3 <script>.py`).

| Script | Produces |
|--------|----------|
| `detection_timeline.py` | `clogging_detection_timeline.png` |
| `five_experiments_figure.py` | `five_experiments_figure.png` (+ cached results) |
| `exp3_7_figure.py` | `exp3_7_figure.png` |
| `generate_background_figures.py` | Background method figures (`fig1`–`fig4`) |
| `generate_spectral_evolution_figure.py` | Spectrogram + spectral-snapshot figures |
| `sigma_sweep.py` | `sigma_sweep_results.csv` (raw sweep data) |
| `sigma_analysis.py` | `sigma_analysis.png`, `sigma_table.tsv` |
| `sigma_performance.py` | `sigma_performance.png` (lead time vs σ) |
| `sigma_signal_quality.py` | `sigma_signal_quality.png`, `sigma_eta_accuracy.png` |
| `sigma_histogram.py` | `sigma_histogram.png` |
| `benchmarks_nfr.py` | Performance report (console output) |
| `split_2906_by_run.py` | Per-run split table (console output) |

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI (Python) |
| Signal processing | NumPy, SciPy |
| ML models | scikit-learn, PyTorch, TensorFlow |
| Frontend | React 19 + TypeScript + Vite |
| State management | Zustand |
| Charts | Recharts |
| Real-time streaming | WebSocket |

---

## Thesis context

This platform is the primary software artifact for a master's thesis investigating whether signal-processing methods can detect partial pipeline clogging earlier and more reliably than traditional pressure-based approaches. The work follows the Design Science Research methodology (Peffers et al.).

**Key confirmed finding:** on a 4h21m multiphase-flow recording, the composite spectral-distance method (FFT-based) detected a partial clogging event at ~1h06m, while the static pressure-deviation method never triggered during the entire recording. This supports the thesis claim that spectral analysis provides significantly earlier warning of partial blockages than pressure monitoring alone.
