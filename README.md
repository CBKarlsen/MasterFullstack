# Pipe Clogging Detection Platform

A real-time monitoring and analysis platform for detecting and predicting clogging in oil pipelines. Built as a master's thesis software artifact at HVL (Western Norway University of Applied Sciences) by Casper Benjamin Karlsen.

---

## What Problem Does This Solve?

In oil and gas pipelines, partial blockages — sand deposits, wax build-up, hydrate formation — can go undetected for hours before causing a full shutdown. By the time a simple pressure sensor triggers an alarm, the blockage is often already severe.

This platform detects clogging much earlier, by listening to the *texture* of the flow rather than just the pressure level. It can identify that something is changing in the pipe long before a traditional pressure alarm would fire.

---

## How It Works — The Big Picture

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
│  4 Detection Methods (run in parallel) │
│  • Static       • Composite (FFT)     │
│  • Turbulence   • Spectral Slope      │
└─────────┬─────────────────────────────┘
          │
          ▼
┌───────────────────┐
│  ML Ensemble      │  Optional: trained models vote on
│  (optional)       │  whether clogging is occurring
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Dashboard        │  Live charts, traffic light,
│  & Forecast       │  and time-to-critical prediction
└───────────────────┘
```

The platform runs in two modes:
- **Live mode**: streams sensor data in real time via WebSocket, updating the dashboard at ~20 Hz
- **Batch mode**: processes an entire file at once — useful for post-hoc analysis, parameter tuning, and thesis evaluation

---

## What Is FFT and Why Does It Matter?

### Sound as an analogy

Imagine you're listening to a musical chord through a wall. You can't see what's happening on the other side, but you can hear *how the sound changes*. If the instruments suddenly sound different, maybe a new buzzing frequency appears, or a familiar tone goes quiet, something has changed, even if you can't see it directly.

Pipelines are similar. When fluid flows through a healthy pipe, it creates a characteristic pattern of vibrations and pressure fluctuations. This pattern has a kind of "signature", specific frequencies are strong, others are weak. When a blockage starts forming, the flow is disrupted: turbulence changes, new vortices appear around the obstruction, and the pressure fluctuation signature shifts.

**FFT (Fast Fourier Transform)** is the mathematical tool that reads this signature.

### What FFT actually does

A pressure sensor records one number every 0.05 seconds (at 20 Hz): `[0.412, 0.415, 0.409, 0.421, ...]`. This is called the **time-domain signal**, it shows you how pressure changes over time, but it's hard to interpret directly because it's just a noisy wiggly line.

FFT transforms that wiggly line into a **frequency breakdown**: it answers the question *"how much of this signal is slow oscillation, and how much is fast oscillation?"*

Think of it like a prism splitting white light into a rainbow. The time-domain signal is the white light. FFT splits it into its component frequencies — the "rainbow" of the pressure signal.

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

During the **calibration phase**, the platform records the FFT spectrum of healthy flow, the "fingerprint" of a normal, unclogged pipe. This is stored as a reference.

During the **analysis phase**, it continuously computes the FFT spectrum of the current signal and asks: *how different is this from the healthy fingerprint?*

This difference is measured as an **L1 distance** , essentially, the total area between the two spectrum shapes. A small distance means flow looks healthy. A large, sustained distance means the spectral character of the flow has changed — something is wrong inside the pipe.

This is what the platform calls the **Composite Score**, and it is the most sensitive detection method available.

### Why this is better than just watching pressure

A simple pressure alarm only fires when the *average* pressure drop changes by a large amount — which only happens when a blockage is severe. The composite method (FFT comparison) detects changes in the *shape* of the pressure fluctuation spectrum, which shifts much earlier — sometimes hours before a pressure alarm would trigger.

**Confirmed result on a real 4h21m dataset:** the composite method detected a partial blockage event at ~1h06m. The traditional pressure-based method never triggered at all during the entire recording.

---

## The Four Detection Methods

Each method looks at a different physical aspect of the flow. They are designed to complement each other — no single method is perfect, but together they provide strong evidence.

### 1. Static — Hydraulic Head Deviation

**What it measures:** The change in average pressure drop compared to the healthy baseline.

**Physics:** A blockage physically restricts the pipe. As the restriction grows, more pressure is needed to push the same amount of fluid through (Hagen-Poiseuille law). The static score measures how much the average pressure drop has shifted from baseline.

**Strengths:** Highly interpretable, directly physical.
**Weakness:** Only fires on significant, sustained mean-level changes. A partial blockage that changes flow texture without dramatically shifting the average pressure will not trigger this method.

---

### 2. Composite — L1 Spectral Distance (FFT-based)

**What it measures:** How much the shape of the FFT power spectrum has changed from the healthy baseline.

**Physics:** Turbulence in a healthy pipe follows a predictable energy cascade (the Kolmogorov cascade — energy flows from large slow eddies to small fast ones in a specific pattern). A partial blockage disrupts this cascade: the obstruction generates new vortex shedding frequencies, shifts turbulent energy toward higher frequencies, and generally alters the spectral "texture" of the flow.

**How it works:**
1. During calibration, the platform computes the FFT of many windows of healthy flow and builds an average "healthy spectrum fingerprint"
2. During analysis, every new window of data is FFT-transformed and compared to this fingerprint
3. The score = the L1 distance between the current normalized spectrum and the baseline fingerprint
4. If the score stays above the threshold for a sustained period, an alert fires

**Strengths:** Sensitive to early-stage partial blockages. Detects changes in flow *texture* long before average pressure shifts.
**Weakness:** Requires a clean calibration period of healthy flow.

---

### 3. Turbulence — Detrended FFT High-Band Energy

**What it measures:** How much energy in the pressure signal is in the high-frequency range (fast oscillations) compared to the low-frequency range (slow oscillations).

**Physics:** A partial blockage acts like a rough obstacle in the flow. Fluid forced around the obstruction breaks into small, fast vortices — increasing high-frequency turbulent energy. This method removes the low-frequency "DC trend" from the signal and measures the fraction of remaining energy at high frequencies.

**Strengths:** Directly reflects turbulence intensity near a blockage.
**Weakness:** Can be sensitive to changes in flow rate that are unrelated to clogging.

---

### 4. Spectral Slope — Power Law Fit

**What it measures:** The steepness of the frequency spectrum on a log-log scale.

**Physics:** In healthy turbulent flow, the Kolmogorov energy cascade produces a power spectrum where energy decreases at a specific rate as frequency increases. On a log-log plot, this looks like a straight line with a slope of approximately −2.5 to −3.0. A clogged pipe with disrupted turbulence has a "flatter" spectrum (slope closer to −1.0 to −1.5) because energy is more evenly distributed across frequencies.

**Strengths:** Provides a physically grounded single number that characterizes the turbulence regime.
**How it's used:** Primarily as an input feature for machine learning models, rather than a standalone threshold detector.

---

## Clogging Trajectory Forecast

Once a detection method crosses its threshold, the platform automatically fits three mathematical growth models to the composite score trajectory after onset, and projects when the score will reach a critical level (default: 2× the detection threshold).

| Model | Equation | Suitable when |
|-------|----------|---------------|
| Linear | `score = a·t + b` | Blockage grows at a constant rate (steady deposition) |
| Exponential | `score = s₀·eᵏᵗ` | Blockage accelerates over time (autocatalytic growth) |
| Power Law | `score = a·tⁿ` | Aggregation kinetics (particle clustering) |

Each model reports its R² (goodness of fit) and estimated time to critical threshold. The consensus ETA is the median across all models that project a future crossing.

**If no model projects a crossing**, the result "No critical crossing projected" indicates the partial blockage has stabilized — the pipe is partially restricted but the restriction is not growing. This is itself a scientifically valuable finding.

---

## Machine Learning Integration

Two ML models are built in:

**Random Forest** (supervised): Trained on labeled data (healthy vs. clogged samples). Takes the 4 detection method scores as input and outputs a clogging probability. Train it in the Models tab by uploading a labeled CSV.

**Isolation Forest** (unsupervised): Trained only on healthy data — no labels needed. Identifies samples that are anomalous compared to normal flow. Useful when you don't have labeled clogging events.

Both models vote in a weighted ensemble. You can tune the influence of each model with the Trust Level slider in the Models tab.

Custom models can be added by dropping `.pkl` (scikit-learn), `.pt` (PyTorch), or `.h5` (Keras) files into the `models/` directory. The platform hot-reloads within 5 seconds.

---

## Getting Started

### Requirements

- Python 3.10+ with a virtual environment
- Node.js 18+

### Backend

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend (port 8000)
python -m app.main
```

### Frontend

```bash
cd frontend
npm install
npm run dev  # Starts on port 3000
```

Open `http://localhost:3000` in your browser. The frontend automatically proxies API calls to the backend.

### Input Data Format

CSV or Excel files with columns (auto-detected, case-insensitive):

| Column | Description | Example name |
|--------|-------------|--------------|
| Time | Timestamp in seconds | `time`, `t` |
| Flow rate | Mass or volumetric flow | `flow`, `flow_rate` |
| Inlet pressure | Pressure at pipe entry | `p_in`, `pressure_in` |
| Outlet pressure | Pressure at pipe exit | `p_out`, `pressure_out` |

Additional columns are passed through to the dashboard as raw data.

---

## Project Structure

```
├── app/
│   ├── main.py          # FastAPI app and all API endpoints
│   ├── engine.py        # WebSocket streaming simulation loop
│   ├── backend.py       # Core signal processing and 4 detection methods
│   ├── batch.py         # Batch analysis (full-file processing)
│   ├── forecast.py      # Clogging trajectory growth model fitting
│   ├── dataloader.py    # CSV/Excel parsing with auto column detection
│   └── models/          # ML model registry, wrappers, and built-in models
├── frontend/
│   └── src/
│       ├── components/  # React UI components
│       ├── store/       # Zustand global state
│       └── hooks/       # WebSocket and data hooks
├── models/              # Drop custom ML model files here (hot-reloaded)
└── data/                # Place sensor data CSV/Excel files here
```

---

## Tech Stack

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

## Thesis Context

This platform is the primary software artifact for a master's thesis investigating whether signal processing methods can detect partial pipeline clogging earlier and more reliably than traditional pressure-based approaches.

**Key confirmed finding:** On a 4h21m multiphase flow recording, the composite spectral distance method (FFT-based) detected a partial clogging event at ~1h06m. The static pressure-deviation method never triggered during the entire recording. This demonstrates that spectral analysis provides significantly earlier warning of partial blockages than traditional pressure monitoring alone.
