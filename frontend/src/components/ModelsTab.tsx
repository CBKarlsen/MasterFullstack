import { useState, useEffect, useCallback, useRef } from "react";
import axios from "axios";
import type { ModelMetadata, DataFile } from "../types";
import { useSimulationStore } from "../store/simulationStore";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function weightLabel(w: number): string {
  if (w <= 0.1) return "Muted — not contributing";
  if (w <= 0.4) return "Low influence";
  if (w <= 0.8) return "Below normal";
  if (w <= 1.2) return "Normal influence";
  if (w <= 1.6) return "High influence";
  if (w <= 2.0) return "Dominant";
  return "Maximum influence";
}

function weightColor(w: number): string {
  if (w <= 0.1) return "#9ca3af";
  if (w <= 0.8) return "#3b82f6";
  if (w <= 1.2) return "#10b981";
  if (w <= 1.6) return "#f59e0b";
  return "#ef4444";
}

const TYPE_LABELS: Record<string, string> = {
  builtin: "Built-in",
  sklearn: "scikit-learn",
  pytorch: "PyTorch",
  tensorflow: "TensorFlow",
  custom: "Custom",
};

const TYPE_COLORS: Record<string, string> = {
  builtin: "#7c3aed",
  sklearn: "#0284c7",
  pytorch: "#ea580c",
  tensorflow: "#16a34a",
  custom: "#6b7280",
};

const SIGMA_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8];

// User-friendly labels mapping to technical hyperparameter values
const N_ESTIMATORS_OPTIONS = [
  { label: "Fast (50 trees)", value: 50 },
  { label: "Balanced (100 trees)", value: 100 },
  { label: "Accurate (200 trees)", value: 200 },
  { label: "Very accurate (300 trees)", value: 300 },
];

const MAX_DEPTH_OPTIONS = [
  { label: "Simple (depth 4)", value: 4 },
  { label: "Balanced (depth 8)", value: 8 },
  { label: "Complex (depth 12)", value: 12 },
  { label: "Unlimited", value: null },
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TrainingInfo {
  source: string;
  is_user_trained: boolean;
  stats?: {
    n_healthy: number;
    n_clogged: number;
    train_accuracy: number;
    test_accuracy: number;
    n_estimators: number;
    max_depth: number | null;
    feature_importances: Record<string, number>;
  } | null;
}

interface TrainingResult {
  message: string;
  file_used: string;
  sigma: number;
  n_healthy: number;
  n_clogged: number;
  train_accuracy: number;
  test_accuracy: number;
  feature_importances: Record<string, number>;
}

// ---------------------------------------------------------------------------
// RandomForest Training Panel
// ---------------------------------------------------------------------------

interface TrainingProgress {
  phase: string;
  percent: number;
  message: string;
  result?: TrainingResult | null;
  error?: string | null;
}

const PHASE_LABELS: Record<string, string> = {
  starting:   "Starting…",
  analyzing:  "Analyzing data…",
  extracting: "Extracting features…",
  training:   "Training model…",
  complete:   "Complete",
  error:      "Failed",
};

function RandomForestTrainingPanel({ onTrainComplete }: { onTrainComplete: () => void }) {
  const [files, setFiles] = useState<DataFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [sigma, setSigma] = useState(3);
  const [calibrationSeconds, setCalibrationSeconds] = useState(30);
  const [includeWarnings, setIncludeWarnings] = useState(false);
  const [nEstimators, setNEstimators] = useState(100);
  const [maxDepth, setMaxDepth] = useState<number | null>(8);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [trainingInfo, setTrainingInfo] = useState<TrainingInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resetting, setResetting] = useState(false);
  const [open, setOpen] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch available files and current training info
  useEffect(() => {
    axios.get<DataFile[]>("/api/data").then((r) => setFiles(r.data)).catch(() => {});
    axios.get<TrainingInfo>("/api/models/random_forest/training-info")
      .then((r) => setTrainingInfo(r.data))
      .catch(() => {});
  }, []);

  // Poll for training progress while training is active
  useEffect(() => {
    if (!training) return;

    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get<TrainingProgress>("/api/models/random_forest/training-progress");
        const state = res.data;
        setProgress(state);

        if (state.phase === "complete" && state.result) {
          clearInterval(pollRef.current!);
          setResult(state.result);
          setTrainingInfo({
            source: `user data: ${selectedFile}`,
            is_user_trained: true,
            stats: state.result as TrainingResult & { feature_importances: Record<string, number> },
          });
          setTraining(false);
          onTrainComplete();
        } else if (state.phase === "error") {
          clearInterval(pollRef.current!);
          setError(state.error ?? "Training failed");
          setTraining(false);
        }
      } catch {
        /* transient poll failure — keep trying */
      }
    }, 600);

    return () => { if (pollRef.current) clearInterval(pollRef.current!); };
  }, [training, selectedFile, onTrainComplete]);

  const handleTrain = async () => {
    if (!selectedFile) return;
    setTraining(true);
    setError(null);
    setResult(null);
    setProgress({ phase: "starting", percent: 2, message: "Starting…" });

    try {
      const params = new URLSearchParams({
        file: selectedFile,
        sigma: String(sigma),
        calibration_seconds: String(calibrationSeconds),
        include_warnings: String(includeWarnings),
        n_estimators: String(nEstimators),
        ...(maxDepth !== null ? { max_depth: String(maxDepth) } : {}),
      });
      // Non-blocking — returns immediately, polling takes over
      await axios.post(`/api/models/random_forest/train?${params}`);
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string } }; message?: string };
      setError(e.response?.data?.detail ?? e.message ?? "Could not start training");
      setTraining(false);
      setProgress(null);
    }
  };

  const handleReset = async () => {
    setResetting(true);
    setError(null);
    try {
      await axios.post("/api/models/random_forest/reset");
      setResult(null);
      setTrainingInfo({ source: "synthetic data", is_user_trained: false, stats: null });
      onTrainComplete();
    } catch {
      setError("Reset failed");
    } finally {
      setResetting(false);
    }
  };

  const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

  return (
    <div style={{ marginTop: "18px", borderTop: "1px solid #e5e7eb", paddingTop: "18px" }}>
      {/* ── Current training status ── */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
        <div>
          <span style={{ fontSize: "13px", fontWeight: 600, color: "#374151" }}>Training data: </span>
          <span style={{
            fontSize: "13px",
            color: trainingInfo?.is_user_trained ? "#16a34a" : "#6b7280",
            fontWeight: trainingInfo?.is_user_trained ? 600 : 400,
          }}>
            {trainingInfo?.source ?? "loading…"}
          </span>
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          {trainingInfo?.is_user_trained && (
            <button
              onClick={handleReset}
              disabled={resetting}
              style={{
                padding: "5px 12px", borderRadius: "6px", border: "1px solid #e5e7eb",
                background: "#fff", fontSize: "12px", cursor: resetting ? "wait" : "pointer", color: "#6b7280",
              }}
            >
              {resetting ? "Resetting…" : "Reset to default"}
            </button>
          )}
          <button
            onClick={() => setOpen((o) => !o)}
            style={{
              padding: "5px 14px", borderRadius: "6px", border: "none",
              background: open ? "#e0e7ff" : "#6366f1", color: open ? "#4338ca" : "#fff",
              fontSize: "12px", fontWeight: 600, cursor: "pointer",
            }}
          >
            {open ? "Close" : "Train on your data"}
          </button>
        </div>
      </div>

      {/* ── Previous training stats ── */}
      {trainingInfo?.is_user_trained && trainingInfo.stats && !open && (
        <div style={{
          padding: "10px 14px", backgroundColor: "#f0fdf4", border: "1px solid #bbf7d0",
          borderRadius: "8px", marginBottom: "12px", fontSize: "12px",
        }}>
          <div style={{ display: "flex", gap: "24px", flexWrap: "wrap" }}>
            <span>✓ Trained on <strong>{(trainingInfo.stats.n_healthy + trainingInfo.stats.n_clogged).toLocaleString()}</strong> samples</span>
            <span>Healthy: <strong>{trainingInfo.stats.n_healthy.toLocaleString()}</strong></span>
            <span>Clogged: <strong>{trainingInfo.stats.n_clogged.toLocaleString()}</strong></span>
            <span>Hold-out accuracy: <strong style={{ color: trainingInfo.stats.test_accuracy > 0.85 ? "#16a34a" : "#d97706" }}>{pct(trainingInfo.stats.test_accuracy)}</strong></span>
          </div>
        </div>
      )}

      {/* ── Training form ── */}
      {open && (
        <div style={{
          padding: "18px 20px", backgroundColor: "#f9fafb", border: "1px solid #e5e7eb",
          borderRadius: "10px", display: "flex", flexDirection: "column", gap: "16px",
        }}>
          <p style={{ margin: 0, fontSize: "13px", color: "#6b7280", lineHeight: 1.6 }}>
            Pick a <strong>file</strong> or <strong>folder</strong> that covers both a healthy period{" "}
            <em>and</em> a clogging event. Folders combine all CSV/XLSX files inside them — ideal
            when your best data is spread across multiple recordings.
            The system classifies each moment automatically and uses those labels to train the model.
          </p>

          {/* File / folder picker */}
          <div>
            <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "6px" }}>
              Training data
              <span style={{ fontSize: "11px", fontWeight: 400, color: "#9ca3af", marginLeft: "6px" }}>
                file or folder
              </span>
            </label>
            <select
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              style={{
                width: "100%", padding: "8px 10px", borderRadius: "6px",
                border: "1px solid #d1d5db", fontSize: "13px", backgroundColor: "#fff",
              }}
            >
              <option value="">— Select a file or folder —</option>

              {/* Folders — all CSV/XLSX files inside are combined */}
              {files.filter((f) => f.is_folder).length > 0 && (
                <optgroup label="📁 Folders (all files combined)">
                  {files.filter((f) => f.is_folder).map((f) => (
                    <option key={f.name} value={f.name}>
                      {f.name} — {f.file_count ?? "?"} files, {f.size_human}
                    </option>
                  ))}
                </optgroup>
              )}

              {/* Individual files */}
              {files.filter((f) => !f.is_folder).length > 0 && (
                <optgroup label="📄 Individual files">
                  {files.filter((f) => !f.is_folder).map((f) => (
                    <option key={f.name} value={f.name}>
                      {f.name} ({f.size_human})
                    </option>
                  ))}
                </optgroup>
              )}
            </select>

            {/* Contextual hint when a folder is selected */}
            {selectedFile && files.find((f) => f.name === selectedFile)?.is_folder && (
              <div style={{
                marginTop: "6px", padding: "8px 10px", backgroundColor: "#eff6ff",
                border: "1px solid #bfdbfe", borderRadius: "6px", fontSize: "12px", color: "#1d4ed8",
              }}>
                All CSV/XLSX files inside this folder will be concatenated and used as one training dataset.
              </div>
            )}
          </div>

          {/* Parameters row */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
            {/* Sigma */}
            <div>
              <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "6px" }}>
                Sensitivity (σ)
                <span style={{ fontSize: "11px", fontWeight: 400, color: "#9ca3af", marginLeft: "6px" }}>
                  higher = fewer alerts
                </span>
              </label>
              <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                {SIGMA_OPTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => setSigma(s)}
                    style={{
                      padding: "4px 10px", borderRadius: "5px",
                      border: `1px solid ${s === sigma ? "#6366f1" : "#e5e7eb"}`,
                      background: s === sigma ? "#6366f1" : "#fff",
                      color: s === sigma ? "#fff" : "#374151",
                      fontSize: "12px", cursor: "pointer", fontWeight: s === sigma ? 600 : 400,
                    }}
                  >
                    {s}σ
                  </button>
                ))}
              </div>
            </div>

            {/* Calibration seconds */}
            <div>
              <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "6px" }}>
                Calibration window
                <span style={{ fontSize: "11px", fontWeight: 400, color: "#9ca3af", marginLeft: "6px" }}>
                  seconds of healthy baseline
                </span>
              </label>
              <input
                type="number"
                value={calibrationSeconds}
                onChange={(e) => setCalibrationSeconds(Math.max(5, Number(e.target.value)))}
                min={5} max={300} step={5}
                style={{
                  width: "90px", padding: "6px 10px", borderRadius: "6px",
                  border: "1px solid #d1d5db", fontSize: "13px",
                }}
              />
              <span style={{ fontSize: "12px", color: "#6b7280", marginLeft: "6px" }}>s</span>
            </div>
          </div>

          {/* Model complexity */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
            <div>
              <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "6px" }}>
                Speed vs. accuracy
              </label>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                {N_ESTIMATORS_OPTIONS.map((o) => (
                  <label key={o.value} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "12px", cursor: "pointer" }}>
                    <input
                      type="radio"
                      name="n_estimators"
                      checked={nEstimators === o.value}
                      onChange={() => setNEstimators(o.value)}
                    />
                    {o.label}
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "6px" }}>
                Model complexity
              </label>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                {MAX_DEPTH_OPTIONS.map((o) => (
                  <label key={String(o.value)} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "12px", cursor: "pointer" }}>
                    <input
                      type="radio"
                      name="max_depth"
                      checked={maxDepth === o.value}
                      onChange={() => setMaxDepth(o.value)}
                    />
                    {o.label}
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Include warnings toggle */}
          <label style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "13px", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={includeWarnings}
              onChange={(e) => setIncludeWarnings(e.target.checked)}
              style={{ width: "16px", height: "16px" }}
            />
            <span>
              <strong>Include warning-phase points as clogged</strong>
              <span style={{ color: "#6b7280", marginLeft: "6px" }}>
                — gives more training data but adds some noise
              </span>
            </span>
          </label>

          {error && (
            <div style={{
              padding: "10px 14px", backgroundColor: "#fef2f2", border: "1px solid #fca5a5",
              borderRadius: "8px", fontSize: "13px", color: "#dc2626",
            }}>
              {error}
            </div>
          )}

          {/* Progress bar — shown while training */}
          {training && progress && (
            <div style={{
              padding: "14px 16px", backgroundColor: "#f5f3ff", border: "1px solid #c4b5fd",
              borderRadius: "10px",
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                <span style={{ fontSize: "13px", fontWeight: 600, color: "#4c1d95" }}>
                  {PHASE_LABELS[progress.phase] ?? progress.phase}
                </span>
                <span style={{ fontSize: "13px", fontWeight: 700, color: "#6d28d9", fontVariantNumeric: "tabular-nums" }}>
                  {progress.percent}%
                </span>
              </div>
              {/* Track */}
              <div style={{ height: "8px", backgroundColor: "#ddd6fe", borderRadius: "4px", overflow: "hidden" }}>
                <div style={{
                  height: "100%", borderRadius: "4px",
                  backgroundColor: "#7c3aed",
                  width: `${progress.percent}%`,
                  transition: "width 0.5s ease",
                }} />
              </div>
              <div style={{ fontSize: "11px", color: "#7c3aed", marginTop: "6px" }}>
                {progress.message}
              </div>
            </div>
          )}

          {/* Train button */}
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button
              onClick={handleTrain}
              disabled={!selectedFile || training}
              style={{
                padding: "10px 28px", borderRadius: "8px", border: "none",
                background: !selectedFile || training ? "#a5b4fc" : "#6366f1",
                color: "#fff", fontWeight: 700, fontSize: "14px",
                cursor: !selectedFile || training ? "not-allowed" : "pointer",
              }}
            >
              {training ? "Training…" : "Train model"}
            </button>
          </div>

          {/* ── Results ── */}
          {result && (
            <div style={{
              padding: "16px", backgroundColor: "#f0fdf4", border: "1px solid #86efac",
              borderRadius: "10px",
            }}>
              <div style={{ fontWeight: 700, fontSize: "14px", color: "#15803d", marginBottom: "12px" }}>
                ✓ Training complete
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px", marginBottom: "12px" }}>
                {[
                  { label: "Healthy samples", value: result.n_healthy.toLocaleString() },
                  { label: "Clogged samples", value: result.n_clogged.toLocaleString() },
                  { label: "Training accuracy", value: pct(result.train_accuracy) },
                  { label: "Hold-out accuracy", value: pct(result.test_accuracy) },
                ].map(({ label, value }) => (
                  <div key={label} style={{
                    padding: "10px", backgroundColor: "#fff", borderRadius: "8px",
                    border: "1px solid #bbf7d0", textAlign: "center",
                  }}>
                    <div style={{ fontSize: "11px", color: "#6b7280", marginBottom: "2px" }}>{label}</div>
                    <div style={{ fontSize: "16px", fontWeight: 700, color: "#15803d" }}>{value}</div>
                  </div>
                ))}
              </div>

              {/* Feature importances */}
              <div>
                <div style={{ fontSize: "12px", fontWeight: 600, color: "#374151", marginBottom: "6px" }}>
                  What the model learned to look at:
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "5px" }}>
                  {Object.entries(result.feature_importances)
                    .sort(([, a], [, b]) => b - a)
                    .map(([feature, importance]) => {
                      const LABELS: Record<string, string> = {
                        static_score: "Pressure deviation",
                        composite_score: "Frequency energy ratio",
                        turbulence_score: "Turbulence level",
                        spectral_slope: "Spectral slope",
                      };
                      return (
                        <div key={feature} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                          <span style={{ fontSize: "12px", color: "#6b7280", width: "160px", flexShrink: 0 }}>
                            {LABELS[feature] ?? feature}
                          </span>
                          <div style={{
                            flex: 1, height: "8px", backgroundColor: "#dcfce7", borderRadius: "4px", overflow: "hidden",
                          }}>
                            <div style={{
                              width: `${importance * 100}%`, height: "100%",
                              backgroundColor: "#16a34a", borderRadius: "4px",
                            }} />
                          </div>
                          <span style={{ fontSize: "12px", color: "#374151", width: "36px", textAlign: "right" }}>
                            {pct(importance)}
                          </span>
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Model management card
// ---------------------------------------------------------------------------

interface ModelCardProps {
  model: ModelMetadata;
  onToggle: (name: string, enabled: boolean) => Promise<void>;
  onWeightChange: (name: string, weight: number) => void;
  onTrainComplete: () => void;
}

function ModelManagementCard({ model, onToggle, onWeightChange, onTrainComplete }: ModelCardProps) {
  const [toggling, setToggling] = useState(false);
  const weight = model.weight ?? 1.0;

  const handleToggle = async () => {
    setToggling(true);
    await onToggle(model.name, !model.enabled);
    setToggling(false);
  };

  return (
    <div style={{
      backgroundColor: "#fff",
      border: "1px solid #e5e7eb",
      borderRadius: "12px",
      padding: "20px 24px",
      opacity: model.enabled ? 1 : 0.55,
      transition: "opacity 0.2s, box-shadow 0.2s",
      boxShadow: model.enabled ? "0 1px 4px rgba(0,0,0,0.06)" : "none",
    }}>
      {/* ── Header ── */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "14px" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
            <h3 style={{ margin: 0, fontSize: "16px", fontWeight: 700 }}>{model.name}</h3>
            <span style={{
              fontSize: "11px", fontWeight: 600, padding: "2px 7px", borderRadius: "4px",
              backgroundColor: TYPE_COLORS[model.model_type] + "20",
              color: TYPE_COLORS[model.model_type],
            }}>
              {TYPE_LABELS[model.model_type] ?? model.model_type}
            </span>
          </div>
          {model.description && (
            <p style={{ margin: 0, fontSize: "13px", color: "#6b7280", maxWidth: "480px", lineHeight: 1.5 }}>
              {model.description}
            </p>
          )}
        </div>

        {/* Enable / disable toggle */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
          <span style={{ fontSize: "12px", color: model.enabled ? "#16a34a" : "#9ca3af", fontWeight: 500 }}>
            {model.enabled ? "Active" : "Off"}
          </span>
          <button
            onClick={handleToggle}
            disabled={toggling}
            title={model.enabled ? "Disable model" : "Enable model"}
            style={{
              width: "42px", height: "24px", borderRadius: "12px", border: "none",
              cursor: toggling ? "wait" : "pointer",
              backgroundColor: model.enabled ? "#16a34a" : "#d1d5db",
              position: "relative", transition: "background-color 0.2s", flexShrink: 0,
            }}
          >
            <span style={{
              position: "absolute", top: "3px", left: model.enabled ? "20px" : "3px",
              width: "18px", height: "18px", borderRadius: "50%", backgroundColor: "#fff",
              transition: "left 0.2s", boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
            }} />
          </button>
        </div>
      </div>

      {/* ── Stats row ── */}
      <div style={{ display: "flex", gap: "24px", marginBottom: "18px", fontSize: "12px", color: "#6b7280" }}>
        {(model.prediction_count ?? 0) > 0 && (
          <span><strong style={{ color: "#374151" }}>{model.prediction_count!.toLocaleString()}</strong> predictions</span>
        )}
        {(model.avg_inference_ms ?? 0) > 0 && (
          <span>Avg speed: <strong style={{ color: "#374151" }}>{model.avg_inference_ms!.toFixed(1)} ms</strong></span>
        )}
        {model.input_type === "sequence" && model.sequence_length && (
          <span>Window: <strong style={{ color: "#374151" }}>{model.sequence_length} steps</strong></span>
        )}
        <span>v{model.version} · {model.author}</span>
      </div>

      {model.last_error && (
        <div style={{
          marginBottom: "14px", padding: "8px 12px", backgroundColor: "#fef2f2",
          border: "1px solid #fca5a5", borderRadius: "6px", fontSize: "12px", color: "#dc2626",
        }}>
          ⚠ Last error: {model.last_error}
        </div>
      )}

      {/* ── Trust level slider ── */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "6px" }}>
          <label style={{ fontSize: "13px", fontWeight: 600, color: "#374151" }}>Trust level</label>
          <span style={{ fontSize: "12px", fontWeight: 600, color: weightColor(weight) }}>
            {weightLabel(weight)}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ fontSize: "11px", color: "#9ca3af", width: "28px" }}>Low</span>
          <input
            type="range" min={0} max={2} step={0.05} value={weight}
            disabled={!model.enabled}
            onChange={(e) => onWeightChange(model.name, parseFloat(e.target.value))}
            style={{ flex: 1, accentColor: weightColor(weight), cursor: model.enabled ? "pointer" : "not-allowed" }}
          />
          <span style={{ fontSize: "11px", color: "#9ca3af", width: "28px", textAlign: "right" }}>High</span>
        </div>
        <p style={{ margin: "6px 0 0", fontSize: "11px", color: "#9ca3af" }}>
          Adjusts how much this model's opinion counts in the final clogging score.
          Setting it to zero silences the model without turning it off.
        </p>
      </div>

      {/* ── Training panel (Random Forest only) ── */}
      {model.name === "Random Forest" && (
        <RandomForestTrainingPanel onTrainComplete={onTrainComplete} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main ModelsTab
// ---------------------------------------------------------------------------

export function ModelsTab() {
  const { models, setModels, toggleModel, updateModelWeight } = useSimulationStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRefs = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get<ModelMetadata[]>("/api/models");
      setModels(res.data);
    } catch {
      setError("Could not load models. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, [setModels]);

  useEffect(() => {
    fetchModels();
    const interval = setInterval(fetchModels, 5000);
    return () => clearInterval(interval);
  }, [fetchModels]);

  const handleToggle = useCallback(async (name: string, enabled: boolean) => {
    try {
      await axios.put(`/api/models/${encodeURIComponent(name)}/enable?enabled=${enabled}`);
      toggleModel(name, enabled);
    } catch {
      setError(`Failed to ${enabled ? "enable" : "disable"} "${name}".`);
    }
  }, [toggleModel]);

  const handleWeightChange = useCallback((name: string, weight: number) => {
    updateModelWeight(name, weight);
    clearTimeout(debounceRefs.current[name]);
    debounceRefs.current[name] = setTimeout(async () => {
      try {
        await axios.put(`/api/models/${encodeURIComponent(name)}/weight?weight=${weight}`);
      } catch {
        setError(`Failed to update weight for "${name}".`);
      }
    }, 300);
  }, [updateModelWeight]);

  const activeCount = models.filter((m) => m.enabled).length;

  return (
    <div style={{ maxWidth: "860px", margin: "0 auto" }}>
      {/* ── Header ── */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "24px" }}>
        <div>
          <h2 style={{ margin: "0 0 4px", fontSize: "20px", fontWeight: 700 }}>Model Management</h2>
          <p style={{ margin: 0, fontSize: "13px", color: "#6b7280" }}>
            {activeCount} of {models.length} model{models.length !== 1 ? "s" : ""} active ·
            Use the Trust level slider to control each model's influence on the clogging score.
          </p>
        </div>
        <button
          onClick={fetchModels}
          disabled={loading}
          style={{
            padding: "7px 14px", borderRadius: "6px", border: "1px solid #e5e7eb",
            background: "#fff", fontSize: "13px", cursor: loading ? "wait" : "pointer", color: "#374151",
          }}
        >
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {error && (
        <div style={{
          marginBottom: "16px", padding: "10px 14px", backgroundColor: "#fef2f2",
          border: "1px solid #fca5a5", borderRadius: "8px", fontSize: "13px", color: "#dc2626",
        }}>
          {error}
        </div>
      )}

      <div style={{
        marginBottom: "20px", padding: "14px 18px", backgroundColor: "#f0f9ff",
        border: "1px solid #bae6fd", borderRadius: "10px", fontSize: "13px", color: "#0369a1", lineHeight: 1.6,
      }}>
        <strong>How the overall score works:</strong> Each active model votes on whether the pipe is
        clogging. The final score is a weighted average of all votes. Increase a model's Trust level
        if it performs well on your data; decrease it if it gives too many false alarms.
      </div>

      {loading && models.length === 0 ? (
        <div style={{ textAlign: "center", padding: "60px", color: "#9ca3af" }}>Loading models…</div>
      ) : models.length === 0 ? (
        <div style={{ textAlign: "center", padding: "60px", color: "#9ca3af" }}>
          No models loaded yet. Upload a model file or restart the backend.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {models.map((model) => (
            <ModelManagementCard
              key={model.name}
              model={model}
              onToggle={handleToggle}
              onWeightChange={handleWeightChange}
              onTrainComplete={fetchModels}
            />
          ))}
        </div>
      )}
    </div>
  );
}
