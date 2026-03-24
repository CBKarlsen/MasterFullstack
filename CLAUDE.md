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

---

<!-- Generated by init-claude-rules | https://github.com/lifedever/claude-rules -->

# Core Development Principles

## Attitude Toward Legacy Code

This is the most important rule: **Do not mimic the style and patterns of existing code in the project.** Always follow this specification.

- When modifying old code, refactor the parts you touch according to this specification. Do not perpetuate bad habits for the sake of "consistency"
- If old code has obvious design problems (God Class, deep nesting, hardcoding, excessive coupling), fix them while making changes
- Do not be afraid to change the structure of old code, as long as behavior remains unchanged
- If the refactoring scope is too large (cascading changes across more than 3 files), explain the plan before proceeding

## Hard Requirements for Code Quality

- A single function must not exceed 30 lines (excluding blank lines and comments); split if it does
- A single file must not exceed 300 lines; split by responsibility if it does
- Nesting depth must not exceed 3 levels (if/for/callback); reduce with early returns, extracted functions, etc.
- Function parameters must not exceed 4; use an object parameter if more are needed
- No commented-out code allowed; delete unused code instead of commenting it out
- No magic numbers or magic strings; extract them into named constants

## Naming

- Names must be semantic; the purpose should be clear from the name alone
- No meaningless names: `data1`, `temp`, `info`, `obj`, `result`, `item` (except loop variables)
- Boolean values use `is`/`has`/`can`/`should` prefixes: `isLoading`, `hasPermission`
- Function names start with a verb: `fetchUser`, `validateInput`, `calculateTotal`
- Constants in ALL_CAPS_SNAKE_CASE: `MAX_RETRY_COUNT`, `API_BASE_URL`
- Event handler functions use `handle` prefix: `handleClick`, `handleSubmit`

## Architecture Principles

- **Single Responsibility**: One function does one thing, one file owns one domain
- **Separation of Concerns**: UI contains no business logic, business logic contains no UI code, data access is a separate layer
- **Unidirectional Dependencies**: Upper layers depend on lower layers, never the reverse. UI -> Business Logic -> Data Layer
- **Program to Interfaces**: Modules communicate through interfaces/protocols, not concrete implementations
- **Composition Over Inheritance**: Use composition unless there is a clear is-a relationship

## Error Handling

- Perform defensive validation only at system boundaries (user input, external API responses, file I/O)
- Internal function calls trust parameter types; no redundant validation
- Error messages should be human-friendly and include context (which operation failed, what values were passed)
- Async operations must have error handling; no bare Promises or unhandled async calls
- Do not wrap the entire function body in try-catch; only wrap the specific operations that may fail

## Avoid Over-Engineering

- Solve only the current problem; do not add abstractions for hypothetical future requirements
- Three lines of duplicated code are better than a premature abstraction
- Do not create utility functions for logic that is used only once
- Do not add unnecessary intermediate layers, wrappers, or adapters
- Add configuration and options only when flexibility is genuinely needed

## Output Requirements

- Always respond in English
- Get straight to the point; no pleasantries or preamble
- Only output information directly relevant to the current task; do not repeat what the user has already said

---

# Git Conventions

## Commit Rules

- Do not commit code automatically unless explicitly requested
- Ensure the code runs correctly before committing
- Commit directly to main/master to stay agile

## Commit Message Format

```
<type>(<scope>): <subject>
```

A space follows the colon. Type values:

| type | Purpose |
|------|---------|
| feat | New feature |
| fix | Bug fix |
| docs | Documentation or comments |
| style | Code formatting (no runtime impact) |
| refactor | Refactoring (not a new feature or bug fix) |
| perf | Performance optimization |
| test | Adding tests |
| chore | Build process or tooling changes |

Use a list when there are more than two key points:

```
feat(web): implement email verification workflow

- Add email verification token generation service
- Create verification email template with dynamic links
- Add API endpoint for token validation
```

---

# TypeScript Guidelines

## Type System

- No `any`. Use `unknown` when the type is uncertain, then narrow with type guards
- Use `interface` for object shapes; use `type` for unions / intersections / mapped types
- Public functions must have explicit return types; internal functions may rely on inference
- Mark properties and parameters that won't be mutated with `readonly`
- Leverage built-in utility types: `Partial<T>`, `Pick<T, K>`, `Omit<T, K>`, `Record<K, V>`
- Generic parameter names should be meaningful: `TItem` rather than bare `T` (single generic parameter excepted)

```typescript
// Forbidden
function parse(data: any): any { ... }

// Correct
function parse(data: unknown): ParseResult { ... }
```

## Naming

- Types and interfaces: `PascalCase` (`UserProfile`, `ApiResponse`)
- Variables and functions: `camelCase` (`getUserById`, `isValid`)
- Constants: `UPPER_CASE` (`MAX_RETRY_COUNT`)
- Enum members: `PascalCase` (`Status.Active`)
- Generic parameters: single uppercase letter or `T` prefix (`T`, `TKey`, `TValue`)

## Module Organization

- One primary export per file (a component, a class, or a group of closely related functions)
- Place type definitions at the top of the file that uses them; shared cross-file types go in a `types/` directory
- Do not use `index.ts` barrel exports -- they cause circular dependencies and tree-shaking issues; import directly from source files
- Separate `import type` from value imports

```typescript
import type { UserProfile } from './types/user'
import { formatDate } from './utils/date'
```

## Functions

- Prefer arrow functions; use `function` only when `this` binding is needed
- Prefer `async/await`; do not chain more than 2 levels of `.then()`
- Handle errors with specific types; do not `catch(e: any)`

```typescript
// Forbidden
fetchData().then(res => process(res)).then(data => save(data)).catch(e => console.log(e))

// Correct
try {
  const res = await fetchData()
  const data = process(res)
  await save(data)
} catch (error) {
  if (error instanceof NetworkError) {
    showNetworkError(error.message)
  }
  throw error
}
```

## Prohibited Patterns

- No `// @ts-ignore` or `// @ts-expect-error` (unless accompanied by a comment explaining why)
- No `as` type assertions (unless narrowing from `unknown` with good reason)
- No `!` non-null assertions (use optional chaining `?.` or early null checks instead)
- No `enum` (use `as const` objects or union types instead to avoid runtime overhead)

```typescript
// Forbidden
enum Status { Active, Inactive }

// Correct
const Status = { Active: 'active', Inactive: 'inactive' } as const
type Status = typeof Status[keyof typeof Status]
```

---

# React Guidelines

## Basic Component Rules

- Use only function components; class components are forbidden
- A single component file must not exceed 200 lines
- Components are responsible only for UI; extract business logic into custom hooks
- Complex expressions in JSX are forbidden; extract them into variables or functions
- A file should export only one component (except small helper components)

## State Management

- Use `useState` for component-local state
- Use `useReducer` for complex state logic
- Keep state as close to where it is used as possible; do not lift state unnecessarily
- Use Context (for small amounts of global state) or Zustand/Jotai (for complex scenarios) for cross-component sharing
- Prop drilling beyond 2 levels is forbidden

```tsx
// Forbidden: prop drilling
<GrandParent user={user}>
  <Parent user={user}>
    <Child user={user} />  // 3 levels deep
  </Parent>
</GrandParent>

// Correct: Context
const UserContext = createContext<User | null>(null)
const useUser = () => {
  const user = useContext(UserContext)
  if (!user) throw new Error('useUser must be used within UserProvider')
  return user
}
```

## Hook Rules

- Custom hook file names must have the `use` prefix: `useAuth.ts`
- A hook should do one thing only
- `useEffect` must have a correct dependency array; suppressing with `// eslint-disable-next-line` is forbidden
- `useEffect` with side effects must return a cleanup function
- Passing an async function directly to `useEffect` is forbidden

```typescript
// Forbidden
useEffect(async () => {
  const data = await fetchData()
  setData(data)
}, [])

// Correct
useEffect(() => {
  const controller = new AbortController()
  const load = async () => {
    try {
      const data = await fetchData({ signal: controller.signal })
      setData(data)
    } catch (error) {
      if (!controller.signal.aborted) setError(error)
    }
  }
  load()
  return () => controller.abort()
}, [])
```

## Performance

- Use `React.memo` only on components with actual performance issues; do not use it preemptively
- Use `useMemo` / `useCallback` only in the following scenarios:
  - Computationally expensive derived values
  - Dependencies of other hooks
  - Props passed to children wrapped with `React.memo`
- Lists must have stable, unique `key` values; using index is forbidden
- Use virtualization for large lists (`react-virtual` / `react-window`)

## Props

- Define with TypeScript interfaces, named `XxxProps`
- Prefer primitive types over objects for props
- Name callback props with `onXxx`: `onClick`, `onSubmit`

```typescript
interface UserCardProps {
  name: string
  email: string
  onEdit: (id: string) => void
}
```

## Error Handling

- Page-level components must have an Error Boundary
- Async operations must handle loading / error / empty states
- Error messages should be user-friendly; log raw errors to the console

## Styling

- Prefer Tailwind CSS
- Use CSS Modules or `clsx`/`cn` for class name concatenation when dynamic styles are needed
- Inline style objects are forbidden (unless the values are truly dynamically computed)
- `!important` is forbidden

---

# Python Guidelines

## Core Principles

- Follow PEP 8; use ruff for formatting and linting
- Type annotations: all public function parameters and return values must have type annotations
- Use `pathlib.Path` instead of `os.path`
- Use f-strings instead of `format()` and `%`

## Naming

- Classes: `PascalCase` (`UserService`, `DataProcessor`)
- Functions and variables: `snake_case` (`get_user_by_id`, `is_valid`)
- Constants: `UPPER_SNAKE_CASE` (`MAX_RETRY_COUNT`)
- Private members: single underscore prefix `_internal_method`
- No double-underscore name mangling (`__private`) unless there is a clear reason

## Type Annotations

```python
# Forbidden
def process(data, config):
    ...

# Correct
def process(data: list[dict[str, Any]], config: ProcessConfig) -> ProcessResult:
    ...
```

- Use `X | None` instead of `Optional[X]` (Python 3.10+)
- Use `list[str]` instead of `List[str]` (Python 3.9+)
- Use `TypeAlias` or `TypedDict` for complex types
- Use `Protocol` to define structural subtypes instead of ABCs

## Error Handling

- Catch specific exceptions; no bare `except:` or `except Exception:`
- Custom business exceptions should inherit from a specific built-in exception class
- Use `raise ... from e` to preserve the exception chain

```python
# Forbidden
try:
    result = call_api()
except:
    pass

# Correct
try:
    result = call_api()
except httpx.TimeoutException as e:
    raise ServiceUnavailableError(f"API timeout: {e.url}") from e
```

## Async

- Use `async/await` for asynchronous code; do not mix threads and coroutines
- Use `asyncio.TaskGroup` for concurrent execution (Python 3.11+)
- Use `contextlib.asynccontextmanager` to manage async resources

## Data Classes

- Use `dataclass` or `pydantic.BaseModel` for simple data containers
- Use `@dataclass(frozen=True)` for immutable data
- Use `pydantic.BaseSettings` for configuration objects

## Project Structure

- Use `pyproject.toml` for project configuration (not `setup.py`)
- Use pytest for testing; place configuration in `[tool.pytest]` within `pyproject.toml`
- Use `uv` or `poetry` for dependency management
