import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from "recharts";
import type { LogEntry } from "../utils/resultLogStore";
import { hhmmssToSeconds } from "../utils/resultLogStore";

// ── Constants ──────────────────────────────────────────────────────────────

const SIGMA_VALUES = [3, 4, 5] as const;
const SIGMA_KEYS: Record<number, keyof LogEntry> = {
  3: "forecastEtaSigma3",
  4: "forecastEtaSigma4",
  5: "forecastEtaSigma5",
};
const FILE_COLORS = [
  "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
  "#0891b2", "#be185d", "#65a30d", "#9333ea", "#0f766e",
  "#c2410c", "#1d4ed8",
];

const DEFAULT_MAX_DURATION_FACTOR = 1.5;
const DEFAULT_MIN_R2 = 0.4;

// ── Types ──────────────────────────────────────────────────────────────────

interface ErrorRow {
  sigma: number;
  [label: string]: number | null;
}

interface RunMeta {
  label: string;
  color: string;
}

interface Filters {
  maxDurationFactor: number;
  minR2: number;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function runLabel(e: LogEntry): string {
  return `${e.filename} (${e.calibrationSeconds}s)`;
}

function toMinutes(hhmmsss: string): number | null {
  const sec = hhmmssToSeconds(hhmmsss);
  return sec !== null ? sec / 60 : null;
}

function isEtaPlausible(etaMin: number, durationMin: number, factor: number): boolean {
  return durationMin <= 0 || etaMin <= durationMin * factor;
}

function getEtaMinutes(entry: LogEntry, sigma: number, filters: Filters): number | null {
  const etaStr = entry[SIGMA_KEYS[sigma]] as string | null;
  if (!etaStr) return null;
  const etaMin = toMinutes(etaStr);
  if (etaMin === null) return null;
  if (!isEtaPlausible(etaMin, entry.durationMin, filters.maxDurationFactor)) return null;
  if (entry.bestFitR2 !== null && entry.bestFitR2 < filters.minR2) return null;
  return etaMin;
}

// ── Filter controls ────────────────────────────────────────────────────────

interface FilterControlsProps {
  filters: Filters;
  onChange: (f: Filters) => void;
  excludedCount: number;
}

function FilterControls({ filters, onChange, excludedCount }: FilterControlsProps) {
  return (
    <div style={{ display: "flex", gap: "20px", alignItems: "center", flexWrap: "wrap",
      padding: "10px 14px", backgroundColor: "#f9fafb", borderRadius: "6px", border: "1px solid #e5e7eb", marginBottom: "12px" }}>
      <span style={{ fontSize: "12px", fontWeight: 600, color: "#374151" }}>Prediction filters:</span>

      <label style={{ fontSize: "12px", color: "#6b7280", display: "flex", alignItems: "center", gap: "6px" }}>
        Max ETA
        <select
          value={filters.maxDurationFactor}
          onChange={(e) => onChange({ ...filters, maxDurationFactor: Number(e.target.value) })}
          style={{ fontSize: "12px", padding: "2px 4px", borderRadius: "4px", border: "1px solid #e5e7eb" }}
        >
          {[1.0, 1.5, 2.0, 3.0, 999].map((v) => (
            <option key={v} value={v}>{v === 999 ? "unlimited" : `${v}× duration`}</option>
          ))}
        </select>
      </label>

      <label style={{ fontSize: "12px", color: "#6b7280", display: "flex", alignItems: "center", gap: "6px" }}>
        Min R²
        <select
          value={filters.minR2}
          onChange={(e) => onChange({ ...filters, minR2: Number(e.target.value) })}
          style={{ fontSize: "12px", padding: "2px 4px", borderRadius: "4px", border: "1px solid #e5e7eb" }}
        >
          {[-999, 0, 0.3, 0.4, 0.5, 0.6, 0.7].map((v) => (
            <option key={v} value={v}>{v === -999 ? "none" : `≥ ${v}`}</option>
          ))}
        </select>
      </label>

      {excludedCount > 0 && (
        <span style={{ fontSize: "12px", color: "#d97706" }}>
          {excludedCount} ETA{excludedCount !== 1 ? "s" : ""} excluded by filters
        </span>
      )}
    </div>
  );
}

// ── Stats summary cards ────────────────────────────────────────────────────

function StatsTable({ errors }: { errors: Record<number, number[]> }) {
  return (
    <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", marginBottom: "16px" }}>
      {SIGMA_VALUES.map((sigma) => {
        const errs = errors[sigma] ?? [];
        if (!errs.length) return (
          <div key={sigma} style={{ padding: "10px 14px", borderRadius: "8px", border: "1px solid #e5e7eb", backgroundColor: "#f9fafb", minWidth: 120 }}>
            <div style={{ fontSize: "13px", fontWeight: 700, marginBottom: "4px" }}>σ = {sigma}</div>
            <div style={{ fontSize: "12px", color: "#9ca3af" }}>No valid predictions</div>
          </div>
        );
        const mae = errs.reduce((s, e) => s + Math.abs(e), 0) / errs.length;
        const bias = errs.reduce((s, e) => s + e, 0) / errs.length;
        return (
          <div key={sigma} style={{ padding: "10px 14px", borderRadius: "8px", border: "1px solid #e5e7eb", backgroundColor: "#f9fafb", minWidth: 120 }}>
            <div style={{ fontSize: "13px", fontWeight: 700, marginBottom: "4px" }}>σ = {sigma}</div>
            <div style={{ fontSize: "12px", color: "#6b7280" }}>MAE: <strong style={{ color: "#374151" }}>{mae.toFixed(1)} min</strong></div>
            <div style={{ fontSize: "12px", color: "#6b7280" }}>Bias: <strong style={{ color: bias > 0 ? "#dc2626" : "#16a34a" }}>{bias > 0 ? "+" : ""}{bias.toFixed(1)} min</strong></div>
            <div style={{ fontSize: "11px", color: "#9ca3af", marginTop: "2px" }}>{errs.length} predictions</div>
          </div>
        );
      })}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

interface SigmaErrorChartProps {
  entries: LogEntry[];
}

export function SigmaErrorChart({ entries }: SigmaErrorChartProps) {
  const [filters, setFilters] = useState<Filters>({ maxDurationFactor: DEFAULT_MAX_DURATION_FACTOR, minR2: DEFAULT_MIN_R2 });

  const { chartData, runs, errorsBySigma, excludedCount } = useMemo(() => {
    const withActual = entries.filter((e) => e.actualCloggingTime);

    // Deduplicate by runLabel — keep last entry per label
    const byLabel = new Map<string, LogEntry>();
    for (const e of withActual) byLabel.set(runLabel(e), e);

    // Count excluded ETAs (exist but fail filters)
    let excluded = 0;
    for (const entry of byLabel.values()) {
      for (const sigma of SIGMA_VALUES) {
        const etaStr = entry[SIGMA_KEYS[sigma]] as string | null;
        if (etaStr && getEtaMinutes(entry, sigma, filters) === null) excluded++;
      }
    }

    const runList: RunMeta[] = Array.from(byLabel.keys())
      .filter((label) => SIGMA_VALUES.some((s) => getEtaMinutes(byLabel.get(label)!, s, filters) !== null))
      .map((label, i) => ({ label, color: FILE_COLORS[i % FILE_COLORS.length] }));

    const errors: Record<number, number[]> = { 3: [], 4: [], 5: [] };
    const rows: ErrorRow[] = SIGMA_VALUES.map((sigma) => {
      const row: ErrorRow = { sigma };
      for (const { label } of runList) {
        const entry = byLabel.get(label)!;
        const actual = toMinutes(entry.actualCloggingTime!);
        const etaMin = getEtaMinutes(entry, sigma, filters);
        if (actual !== null && etaMin !== null) {
          const err = etaMin - actual;
          row[label] = err;
          errors[sigma].push(err);
        }
      }
      return row;
    });

    return { chartData: rows, runs: runList, errorsBySigma: errors, excludedCount: excluded };
  }, [entries, filters]);

  if (!entries.some((e) => e.actualCloggingTime)) {
    return (
      <div style={{ padding: "20px", textAlign: "center", color: "#9ca3af", fontSize: "13px",
        border: "1px dashed #e5e7eb", borderRadius: "8px", marginTop: "16px" }}>
        No data yet — enter actual clogging times in the table to populate this chart.
      </div>
    );
  }

  return (
    <div style={{ marginTop: "20px" }}>
      <div style={{ fontSize: "13px", fontWeight: 600, color: "#374151", marginBottom: "8px" }}>
        Prediction Error per σ
        <span style={{ fontWeight: 400, color: "#6b7280", marginLeft: "8px" }}>
          positive = predicted too late · negative = predicted too early · zero = perfect
        </span>
      </div>

      <FilterControls filters={filters} onChange={setFilters} excludedCount={excludedCount} />
      <StatsTable errors={errorsBySigma} />

      {runs.length === 0 ? (
        <div style={{ padding: "16px", textAlign: "center", color: "#9ca3af", fontSize: "13px",
          border: "1px dashed #e5e7eb", borderRadius: "8px" }}>
          All predictions excluded by current filters — try relaxing the R² or duration limits.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData} margin={{ top: 8, right: 32, left: 16, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="sigma" type="number" domain={[2.8, 5.2]} ticks={[3, 4, 5]}
              tickFormatter={(v) => `σ=${v}`} stroke="#9ca3af" fontSize={12} />
            <YAxis stroke="#9ca3af" fontSize={11}
              tickFormatter={(v) => `${Number(v) > 0 ? "+" : ""}${Number(v).toFixed(0)} min`}
              label={{ value: "Error (min)", angle: -90, position: "insideLeft", offset: 16, fontSize: 12, fill: "#6b7280" }} />
            <Tooltip
              formatter={(v: unknown, name: string) => {
                const n = Number(v);
                return [`${n > 0 ? "+" : ""}${n.toFixed(1)} min`, name];
              }}
              labelFormatter={(l) => `σ = ${l}`}
            />
            <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
            <ReferenceLine y={0} stroke="#374151" strokeWidth={1.5} strokeDasharray="6 3"
              label={{ value: "Perfect", position: "right", fontSize: 11, fill: "#374151" }} />
            {runs.map(({ label, color }) => (
              <Line key={label} type="monotone" dataKey={label} name={label}
                stroke={color} strokeWidth={2} connectNulls isAnimationActive={false}
                dot={{ r: 5, fill: color, strokeWidth: 2, stroke: "#fff" }} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
