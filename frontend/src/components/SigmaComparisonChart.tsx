import { useState, useMemo, useEffect } from "react";
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
  ScatterChart, Scatter,
} from "recharts";
import type { AnalysisResult } from "./Batchanalysis";
import {
  SIGMA_VALUES,
  computeSigmaForecasts,
  loadStore,
  saveRecord,
  setActualTime,
  clearStore,
  removeRecord,
} from "../utils/sigmaComparisonStore";
import type { SigmaStore } from "../utils/sigmaComparisonStore";

// ── Constants ─────────────────────────────────────────────────────────────

const SIGMA_COLORS: Record<number, string> = { 3: "#3b82f6", 4: "#f97316", 5: "#ef4444" };
const NO_CROSSING_STUB_SEC = 300; // 5-min stub when no ETA is projected

// ── Helpers ────────────────────────────────────────────────────────────────

function formatHHMMSS(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
}

function parseHHMMSS(input: string): number | null {
  const parts = input.split(":").map(Number);
  if (parts.length !== 3 || parts.some(isNaN)) return null;
  return parts[0] * 3600 + parts[1] * 60 + parts[2];
}

// ── Ground-truth input ─────────────────────────────────────────────────────

interface GroundTruthInputProps {
  filename: string;
  currentValue: number | null;
  onSave: (seconds: number) => void;
}

function GroundTruthInput({ filename, currentValue, onSave }: GroundTruthInputProps) {
  const [inputValue, setInputValue] = useState(
    currentValue !== null ? formatHHMMSS(currentValue) : "",
  );
  const isValid = parseHHMMSS(inputValue) !== null;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
      <span style={{ fontSize: "13px", color: "#374151", fontWeight: 500 }}>
        Actual clogging time for <em>{filename}</em>:
      </span>
      <input
        type="text" value={inputValue} onChange={(e) => setInputValue(e.target.value)}
        placeholder="HH:MM:SS"
        style={{
          width: "95px", padding: "4px 8px", borderRadius: "4px", fontFamily: "monospace",
          border: `1px solid ${isValid || !inputValue ? "#e5e7eb" : "#fca5a5"}`, fontSize: "13px",
        }}
      />
      <button
        onClick={() => { const s = parseHHMMSS(inputValue); if (s !== null) onSave(s); }}
        disabled={!isValid}
        style={{
          padding: "4px 12px", borderRadius: "4px", border: "none", fontSize: "12px",
          backgroundColor: isValid ? "#2563eb" : "#d1d5db", color: "#fff",
          cursor: isValid ? "pointer" : "not-allowed",
        }}
      >
        Save
      </button>
      <span style={{ fontSize: "12px", color: "#6b7280" }}>(from video/lab record)</span>
    </div>
  );
}

// ── Timeline chart ─────────────────────────────────────────────────────────

interface TimelineRow {
  label: string;
  sigmaNum: number;
  spacer: number;   // transparent: 0 → onset
  window: number;   // colored: onset → eta (or stub if no crossing)
  stub: number;     // gray stub when no crossing
  onsetTime: number | null;
  consensusEta: number | null;
}

function buildTimelineData(
  sigmaForecasts: ReturnType<typeof computeSigmaForecasts>,
): TimelineRow[] {
  return SIGMA_VALUES.map((sigma) => {
    const { onsetTime, consensusEta } = sigmaForecasts[sigma];
    const onset = onsetTime ?? 0;
    const hasCrossing = consensusEta !== null;
    return {
      label: `σ=${sigma}`,
      sigmaNum: sigma,
      spacer: onset,
      window: hasCrossing ? Math.max(0, consensusEta - onset) : 0,
      stub: hasCrossing ? 0 : NO_CROSSING_STUB_SEC,
      onsetTime,
      consensusEta,
    };
  });
}

interface TimelineTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: TimelineRow }>;
}

function TimelineTooltip({ active, payload }: TimelineTooltipProps) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div style={{ background: "#fff", border: "1px solid #e5e7eb", borderRadius: "6px", padding: "10px 14px", fontSize: "12px" }}>
      <div style={{ fontWeight: 700, marginBottom: "6px" }}>{row.label}</div>
      <div>Detection onset: <strong>{row.onsetTime !== null ? formatHHMMSS(row.onsetTime) : "—"}</strong></div>
      <div>Predicted ETA: <strong style={{ color: row.consensusEta !== null ? "#dc2626" : "#9ca3af" }}>
        {row.consensusEta !== null ? formatHHMMSS(row.consensusEta) : "No crossing projected"}
      </strong></div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export interface SigmaComparisonChartProps {
  result: AnalysisResult;
  selectedFile: string;
  smoothingWindowSec: number;
}

export function SigmaComparisonChart({ result, selectedFile, smoothingWindowSec }: SigmaComparisonChartProps) {
  const displayName = selectedFile.split(/[\\/]/).pop() ?? selectedFile;
  const [store, setStore] = useState<SigmaStore>(() => loadStore());

  const sigmaForecasts = useMemo(
    () => computeSigmaForecasts(result, smoothingWindowSec),
    [result, smoothingWindowSec],
  );

  useEffect(() => { saveRecord(selectedFile, sigmaForecasts); }, [selectedFile, sigmaForecasts]);

  const effectiveStore = useMemo(() => ({
    ...store,
    [selectedFile]: {
      filename: selectedFile,
      actualCloggingTime: store[selectedFile]?.actualCloggingTime ?? null,
      forecasts: sigmaForecasts,
    },
  }), [store, selectedFile, sigmaForecasts]);

  const handleSaveActualTime = (seconds: number) => {
    setActualTime(selectedFile, seconds);
    setStore((prev) => ({
      ...prev,
      [selectedFile]: { ...prev[selectedFile], actualCloggingTime: seconds },
    }));
  };

  const handleClearAll = () => {
    clearStore();
    setStore({});
  };

  const handleRemoveCurrent = () => {
    removeRecord(selectedFile);
    setStore((prev) => { const next = { ...prev }; delete next[selectedFile]; return next; });
  };

  const actualTime = effectiveStore[selectedFile]?.actualCloggingTime ?? null;
  const timelineData = useMemo(() => buildTimelineData(sigmaForecasts), [sigmaForecasts]);

  const storedFileCount = Object.keys(store).length;

  const scatterRecords = Object.values(effectiveStore).filter(
    (r) => r.actualCloggingTime !== null && SIGMA_VALUES.some((s) => r.forecasts[s]?.consensusEta !== null),
  );
  const showScatter = scatterRecords.length >= 1;

  return (
    <div style={{ marginTop: "20px", padding: "20px", backgroundColor: "#fff", borderRadius: "8px", border: "1px solid #e5e7eb" }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "4px" }}>
        <h3 style={{ margin: 0, fontSize: "16px", fontWeight: 700 }}>Sigma Comparison</h3>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <span style={{ fontSize: "12px", color: "#6b7280" }}>
            {storedFileCount} file{storedFileCount !== 1 ? "s" : ""} stored
          </span>
          <button
            onClick={handleRemoveCurrent}
            title="Remove current file's stored data"
            style={{ fontSize: "11px", padding: "2px 8px", borderRadius: "4px", border: "1px solid #e5e7eb", background: "#fff", color: "#6b7280", cursor: "pointer" }}
          >
            Remove this file
          </button>
          <button
            onClick={handleClearAll}
            title="Clear all stored comparison data"
            style={{ fontSize: "11px", padding: "2px 8px", borderRadius: "4px", border: "1px solid #fca5a5", background: "#fff", color: "#dc2626", cursor: "pointer" }}
          >
            Clear all
          </button>
        </div>
      </div>
      <p style={{ margin: "0 0 16px", fontSize: "13px", color: "#6b7280" }}>
        Timeline shows detection onset and predicted clogging time per σ.
        The red line marks the actual (ground-truth) clogging time — enter it below to see accuracy.
      </p>

      <div style={{ marginBottom: "16px", padding: "12px 16px", backgroundColor: "#f9fafb", borderRadius: "6px", border: "1px solid #e5e7eb" }}>
        <GroundTruthInput filename={displayName} currentValue={actualTime} onSave={handleSaveActualTime} />
      </div>

      {/* Timeline chart */}
      <div style={{ fontSize: "13px", fontWeight: 600, color: "#374151", marginBottom: "8px" }}>
        Detection onset → Predicted ETA per σ
        <span style={{ fontWeight: 400, color: "#6b7280", marginLeft: "8px" }}>
          (bar = prediction window; gray stub = no crossing projected)
        </span>
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart layout="vertical" data={timelineData} margin={{ top: 4, right: 24, left: 8, bottom: 4 }} barSize={24}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} />
          <XAxis type="number" tickFormatter={(v) => formatHHMMSS(Number(v))} stroke="#9ca3af" fontSize={10} />
          <YAxis type="category" dataKey="label" stroke="#9ca3af" fontSize={12} width={36} />
          <Tooltip content={<TimelineTooltip />} />
          <Bar dataKey="spacer" stackId="a" fill="transparent" stroke="none" legendType="none" />
          <Bar dataKey="window" stackId="a" name="Prediction window" radius={[0, 3, 3, 0]}>
            {timelineData.map((row) => <Cell key={row.label} fill={SIGMA_COLORS[row.sigmaNum]} fillOpacity={0.85} />)}
          </Bar>
          <Bar dataKey="stub" stackId="a" name="No crossing →" fill="#d1d5db" radius={[0, 3, 3, 0]} legendType="none" />
          {actualTime !== null && (
            <ReferenceLine
              x={actualTime} stroke="#dc2626" strokeWidth={2} strokeDasharray="5 3"
              label={{ value: `Actual: ${formatHHMMSS(actualTime)}`, position: "top", fontSize: 10, fill: "#dc2626" }}
            />
          )}
          <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "4px" }} />
        </BarChart>
      </ResponsiveContainer>

      {/* Multi-file scatter — only once ≥2 files are annotated */}
      {showScatter && (
        <MultiFileScatter store={effectiveStore} />
      )}
      {!showScatter && (
        <p style={{ margin: "12px 0 0", fontSize: "12px", color: "#9ca3af", textAlign: "center" }}>
          Enter the actual clogging time above and click Save to generate the predicted-vs-actual scatter.
        </p>
      )}
    </div>
  );
}

// ── Multi-file scatter (predicted vs actual) ───────────────────────────────

function MultiFileScatter({ store }: { store: SigmaStore }) {
  const seriesData = SIGMA_VALUES.map((sigma) => ({
    sigma,
    points: Object.values(store)
      .filter((r) => r.actualCloggingTime !== null && r.forecasts[sigma]?.consensusEta !== null)
      .map((r) => ({ x: r.forecasts[sigma].consensusEta as number, y: r.actualCloggingTime as number, file: r.filename.split(/[\\/]/).pop() ?? r.filename })),
  }));

  const allVals = seriesData.flatMap((s) => s.points.flatMap((p) => [p.x, p.y]));
  const lo = Math.min(...allVals) - 300;
  const hi = Math.max(...allVals) + 300;

  return (
    <>
      <div style={{ fontSize: "13px", fontWeight: 600, color: "#374151", margin: "20px 0 8px" }}>
        Multi-file: Predicted vs Actual (σ comparison)
        <span style={{ fontWeight: 400, color: "#6b7280", marginLeft: "8px" }}>
          — points on the diagonal = perfect prediction
        </span>
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <ScatterChart margin={{ top: 16, right: 24, left: 16, bottom: 48 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis type="number" dataKey="x" domain={[lo, hi]} tickFormatter={(v) => formatHHMMSS(Number(v))} stroke="#9ca3af" fontSize={10}
            label={{ value: "Predicted clogging time", position: "insideBottom", offset: -30, fontSize: 12, fill: "#6b7280" }} />
          <YAxis type="number" dataKey="y" domain={[lo, hi]} tickFormatter={(v) => formatHHMMSS(Number(v))} stroke="#9ca3af" fontSize={10} width={60}
            label={{ value: "Actual clogging time", angle: -90, position: "insideLeft", offset: 16, fontSize: 12, fill: "#6b7280" }} />
          <Tooltip formatter={(v: unknown, name: string) => [formatHHMMSS(Number(v)), name]} />
          <Legend />
          {/* Diagonal drawn first so dots render on top */}
          <Scatter
            name="y=x (perfect)"
            data={[{ x: lo, y: lo }, { x: hi, y: hi }]}
            line={{ stroke: "#cbd5e1", strokeDasharray: "6 4", strokeWidth: 1 }}
            shape={() => <g />}
            legendType="line"
            fill="none"
          />
          {seriesData.map(({ sigma, points }) => (
            <Scatter
              key={sigma}
              name={`σ=${sigma}`}
              data={points}
              fill={SIGMA_COLORS[sigma]}
              shape={(props: Record<string, unknown>) => {
                const cx = Number(props.cx ?? 0);
                const cy = Number(props.cy ?? 0);
                return (
                  <g>
                    <circle cx={cx} cy={cy} r={8} fill={SIGMA_COLORS[sigma]} stroke="#fff" strokeWidth={2.5} />
                    <text x={cx} y={cy - 12} textAnchor="middle" fontSize={10} fill={SIGMA_COLORS[sigma]} fontWeight={700}>
                      σ={sigma}
                    </text>
                  </g>
                );
              }}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </>
  );
}
