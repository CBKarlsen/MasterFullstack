import { useState, useEffect, useRef } from "react";
import axios from "axios";
import type { DataFile } from "../types";
import type { TrainingProgress } from "./RFTrainingPanel";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface IFTrainingInfo {
  source: string;
  is_user_trained: boolean;
  is_synthetic?: boolean;
  stats?: {
    n_healthy_samples: number;
    n_estimators: number;
    contamination: number;
    flag_rate_on_training: number;
  } | null;
}

interface IFTrainingResult {
  message: string;
  file_used: string;
  sigma: number;
  n_healthy_samples: number;
  n_estimators: number;
  contamination: number;
  flag_rate_on_training: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SIGMA_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8];

const IF_N_ESTIMATORS_OPTIONS = [
  { label: "Fast (50 trees)", value: 50 },
  { label: "Balanced (100 trees)", value: 100 },
  { label: "Accurate (200 trees)", value: 200 },
];

const CONTAMINATION_OPTIONS = [
  { label: "Very clean (1%)", value: 0.01, hint: "Almost no noise in healthy data" },
  { label: "Typical (5%)", value: 0.05, hint: "Recommended for most datasets" },
  { label: "Noisy (10%)", value: 0.10, hint: "Training data has some anomalies mixed in" },
  { label: "Very noisy (20%)", value: 0.20, hint: "Heavily mixed / uncertain healthy data" },
];

const PHASE_LABELS: Record<string, string> = {
  starting:   "Starting…",
  analyzing:  "Analyzing data…",
  extracting: "Extracting features…",
  training:   "Training model…",
  complete:   "Complete",
  error:      "Failed",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface IFTrainingPanelProps {
  onTrainComplete: () => void;
}

export function IFTrainingPanel({ onTrainComplete }: IFTrainingPanelProps) {
  const [files, setFiles] = useState<DataFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [sigma, setSigma] = useState(3);
  const [calibrationSeconds, setCalibrationSeconds] = useState(120);
  const [nEstimators, setNEstimators] = useState(100);
  const [contamination, setContamination] = useState(0.05);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [result, setResult] = useState<IFTrainingResult | null>(null);
  const [trainingInfo, setTrainingInfo] = useState<IFTrainingInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resetting, setResetting] = useState(false);
  const [open, setOpen] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    axios.get<DataFile[]>("/api/data").then((r) => setFiles(r.data)).catch(() => {});
    axios.get<IFTrainingInfo>("/api/models/isolation_forest/training-info")
      .then((r) => setTrainingInfo(r.data))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!training) return;

    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get<TrainingProgress>("/api/models/isolation_forest/training-progress");
        const state = res.data;
        setProgress(state);

        if (state.phase === "complete" && state.result) {
          clearInterval(pollRef.current!);
          setResult(state.result as IFTrainingResult);
          setTrainingInfo({ source: `user data: ${selectedFile}`, is_user_trained: true, stats: state.result as IFTrainingResult });
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
        n_estimators: String(nEstimators),
        contamination: String(contamination),
      });
      await axios.post(`/api/models/isolation_forest/train?${params}`);
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
      await axios.post("/api/models/isolation_forest/reset");
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
              background: open ? "#e0f2fe" : "#0284c7", color: open ? "#0369a1" : "#fff",
              fontSize: "12px", fontWeight: 600, cursor: "pointer",
            }}
          >
            {open ? "Close" : "Train on your data"}
          </button>
        </div>
      </div>

      {(trainingInfo === null || !trainingInfo.is_user_trained) && !open && (
        <div style={{
          padding: "10px 14px", backgroundColor: "#fffbeb", border: "1px solid #fcd34d",
          borderRadius: "8px", marginBottom: "12px", fontSize: "12px", color: "#92400e", lineHeight: 1.6,
        }}>
          <span style={{ fontSize: "14px", marginRight: "6px" }}>⚠️</span>
          <strong>Dormant — not contributing to predictions.</strong>
          {" "}Train it on a healthy recording to activate it.
        </div>
      )}

      {trainingInfo?.is_user_trained && trainingInfo.stats && !open && (
        <div style={{
          padding: "10px 14px", backgroundColor: "#f0fdf4", border: "1px solid #bbf7d0",
          borderRadius: "8px", marginBottom: "12px", fontSize: "12px",
        }}>
          <div style={{ display: "flex", gap: "24px", flexWrap: "wrap" }}>
            <span>✓ Trained on <strong>{trainingInfo.stats.n_healthy_samples.toLocaleString()}</strong> healthy samples</span>
            <span>Trees: <strong>{trainingInfo.stats.n_estimators}</strong></span>
            <span>Flag rate on training: <strong style={{ color: "#d97706" }}>{pct(trainingInfo.stats.flag_rate_on_training)}</strong></span>
          </div>
        </div>
      )}

      {open && (
        <div style={{
          padding: "18px 20px", backgroundColor: "#f9fafb", border: "1px solid #e5e7eb",
          borderRadius: "10px", display: "flex", flexDirection: "column", gap: "16px",
        }}>
          <div style={{
            padding: "10px 14px", backgroundColor: "#f0f9ff", border: "1px solid #bae6fd",
            borderRadius: "8px", fontSize: "12px", color: "#0369a1", lineHeight: 1.6,
          }}>
            <strong>Unsupervised learning:</strong> Point this at recordings where the pipe is <strong>healthy</strong>.
            It learns the normal operating pattern and flags anything that deviates — no clogged labels needed.
          </div>

          <div>
            <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "6px" }}>
              Training data
              <span style={{ fontSize: "11px", fontWeight: 400, color: "#9ca3af", marginLeft: "6px" }}>file or folder — healthy recordings work best</span>
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
              {files.filter((f) => f.is_folder).length > 0 && (
                <optgroup label="📁 Folders (all files combined)">
                  {files.filter((f) => f.is_folder).map((f) => (
                    <option key={f.name} value={f.name}>{f.name} — {f.file_count ?? "?"} files, {f.size_human}</option>
                  ))}
                </optgroup>
              )}
              {files.filter((f) => !f.is_folder).length > 0 && (
                <optgroup label="📄 Individual files">
                  {files.filter((f) => !f.is_folder).map((f) => (
                    <option key={f.name} value={f.name}>{f.name} ({f.size_human})</option>
                  ))}
                </optgroup>
              )}
            </select>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
            <div>
              <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "6px" }}>
                Healthy threshold (σ)
                <span style={{ fontSize: "11px", fontWeight: 400, color: "#9ca3af", marginLeft: "6px" }}>lower = more healthy samples used</span>
              </label>
              <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                {SIGMA_OPTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => setSigma(s)}
                    style={{
                      padding: "4px 10px", borderRadius: "5px",
                      border: `1px solid ${s === sigma ? "#0284c7" : "#e5e7eb"}`,
                      background: s === sigma ? "#0284c7" : "#fff",
                      color: s === sigma ? "#fff" : "#374151",
                      fontSize: "12px", cursor: "pointer", fontWeight: s === sigma ? 600 : 400,
                    }}
                  >{s}σ</button>
                ))}
              </div>
            </div>
            <div>
              <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "6px" }}>
                Calibration window
                <span style={{ fontSize: "11px", fontWeight: 400, color: "#9ca3af", marginLeft: "6px" }}>seconds of healthy baseline</span>
              </label>
              <input
                type="number" value={calibrationSeconds}
                onChange={(e) => setCalibrationSeconds(Math.max(5, Number(e.target.value)))}
                min={5} max={300} step={5}
                style={{ width: "90px", padding: "6px 10px", borderRadius: "6px", border: "1px solid #d1d5db", fontSize: "13px" }}
              />
              <span style={{ fontSize: "12px", color: "#6b7280", marginLeft: "6px" }}>s</span>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
            <div>
              <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "6px" }}>Speed vs. accuracy</label>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                {IF_N_ESTIMATORS_OPTIONS.map((o) => (
                  <label key={o.value} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "12px", cursor: "pointer" }}>
                    <input type="radio" name="if_n_estimators" checked={nEstimators === o.value} onChange={() => setNEstimators(o.value)} />
                    {o.label}
                  </label>
                ))}
              </div>
            </div>
            <div>
              <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "6px" }}>Expected noise in healthy data</label>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                {CONTAMINATION_OPTIONS.map((o) => (
                  <label key={o.value} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "12px", cursor: "pointer" }} title={o.hint}>
                    <input type="radio" name="contamination" checked={contamination === o.value} onChange={() => setContamination(o.value)} />
                    {o.label}
                  </label>
                ))}
              </div>
            </div>
          </div>

          {error && (
            <div style={{ padding: "10px 14px", backgroundColor: "#fef2f2", border: "1px solid #fca5a5", borderRadius: "8px", fontSize: "13px", color: "#dc2626" }}>
              {error}
            </div>
          )}

          {training && progress && (
            <div style={{ padding: "14px 16px", backgroundColor: "#f0f9ff", border: "1px solid #bae6fd", borderRadius: "10px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                <span style={{ fontSize: "13px", fontWeight: 600, color: "#0c4a6e" }}>{PHASE_LABELS[progress.phase] ?? progress.phase}</span>
                <span style={{ fontSize: "13px", fontWeight: 700, color: "#0284c7", fontVariantNumeric: "tabular-nums" }}>{progress.percent}%</span>
              </div>
              <div style={{ height: "8px", backgroundColor: "#bae6fd", borderRadius: "4px", overflow: "hidden" }}>
                <div style={{ height: "100%", borderRadius: "4px", backgroundColor: "#0284c7", width: `${progress.percent}%`, transition: "width 0.5s ease" }} />
              </div>
              <div style={{ fontSize: "11px", color: "#0284c7", marginTop: "6px" }}>{progress.message}</div>
            </div>
          )}

          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button
              onClick={handleTrain}
              disabled={!selectedFile || training}
              style={{
                padding: "10px 28px", borderRadius: "8px", border: "none",
                background: !selectedFile || training ? "#7dd3fc" : "#0284c7",
                color: "#fff", fontWeight: 700, fontSize: "14px",
                cursor: !selectedFile || training ? "not-allowed" : "pointer",
              }}
            >
              {training ? "Training…" : "Train model"}
            </button>
          </div>

          {result && (
            <div style={{ padding: "16px", backgroundColor: "#f0fdf4", border: "1px solid #86efac", borderRadius: "10px" }}>
              <div style={{ fontWeight: 700, fontSize: "14px", color: "#15803d", marginBottom: "12px" }}>✓ Training complete</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px" }}>
                {[
                  { label: "Healthy samples", value: result.n_healthy_samples.toLocaleString() },
                  { label: "Trees", value: String(result.n_estimators) },
                  { label: "Flag rate on training", value: pct(result.flag_rate_on_training) },
                ].map(({ label, value }) => (
                  <div key={label} style={{ padding: "10px", backgroundColor: "#fff", borderRadius: "8px", border: "1px solid #bbf7d0", textAlign: "center" }}>
                    <div style={{ fontSize: "11px", color: "#6b7280", marginBottom: "2px" }}>{label}</div>
                    <div style={{ fontSize: "16px", fontWeight: 700, color: "#15803d" }}>{value}</div>
                  </div>
                ))}
              </div>
              <p style={{ margin: "10px 0 0", fontSize: "12px", color: "#6b7280" }}>
                The model has learned what "normal" looks like and will now flag deviations independently.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
