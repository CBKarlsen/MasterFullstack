import { useState, useMemo, useCallback } from "react";
import axios from "axios";
import {
  ComposedChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
} from "recharts";
import { ControlChart, type ControlChartDataPoint } from "./ControlChart";

interface AnalysisPoint {
  time: number;
  flow: number;
  pressure_drop: number;
  static_score: number;
  composite_score: number;
  turbulence_score: number;
  spectral_slope: number;
  traffic_light: string;
  light_msg: string;
  ensemble_probability: number;
  phase: string;
  raw?: Record<string, number>;
}

interface CalibrationStats {
  baseline_mean: number;
  baseline_std: number;
  composite_mean: number;
  composite_std: number;
  samples_used: number;
}

interface Thresholds {
  sigma: number;
  fft_threshold: number;
  static_threshold: number;
}

interface AnalysisResult {
  timeseries: AnalysisPoint[];
  columns: string[];
  calibration: CalibrationStats;
  thresholds: Thresholds;
  metadata: {
    total_points: number;
    duration_seconds: number;
    sampling_hz: number;
    file: string;
  };
}

interface BatchAnalysisProps {
  selectedFile: string | null;
}

const SIGMA_OPTIONS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

export function BatchAnalysis({ selectedFile }: BatchAnalysisProps) {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sigma, setSigma] = useState(3.0);
  const [thresholds, setThresholds] = useState<Thresholds | null>(null);
  const [calibrationSeconds, setCalibrationSeconds] = useState(30);
  const [smoothingWindowSec, setSmoothingWindowSec] = useState(60);

  // Run full analysis
  const runAnalysis = useCallback(async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post<AnalysisResult>(
        `/api/analyze?file=${encodeURIComponent(selectedFile)}&sigma=${sigma}&calibration_seconds=${calibrationSeconds}`,
      );
      setResult(response.data);
      setThresholds(response.data.thresholds);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  }, [selectedFile, sigma, calibrationSeconds]);

  // Recompute thresholds client-side (instant, no server call needed)
  const handleSigmaChange = useCallback(
    (newSigma: number) => {
      setSigma(newSigma);
      if (!result) return;

      const { composite_mean, composite_std, baseline_std } =
        result.calibration;
      const newFft =
        composite_std > 0
          ? composite_mean + newSigma * composite_std
          : Math.max(0.05, composite_mean * 5.0);
      const newStatic = newSigma * baseline_std;

      setThresholds({
        sigma: newSigma,
        fft_threshold: newFft,
        static_threshold: newStatic,
      });
    },
    [result],
  );

  // Downsample for display (every Nth point to keep charts responsive)
  // Also computes a time-based rolling mean of composite_score for the overlay line.
  // Both composite and compositeSmoothed share the same downsampled timestamps so
  // they can be safely zipped by index in ControlChart.
  const displayData = useMemo(() => {
    if (!result) return { flow: [], static_: [], composite: [], compositeSmoothed: [] };

    const points = result.timeseries;
    const maxDisplay = 3000;
    const step = Math.max(1, Math.floor(points.length / maxDisplay));

    // --- Pass 1: rolling mean over ALL analysis-phase points (full resolution) ---
    const analysisPoints = points.filter((p) => p.phase === "analysis");
    const rollingMeanByTime = new Map<number, number>();
    let wStart = 0;
    let wSum = 0;

    for (let i = 0; i < analysisPoints.length; i++) {
      const p = analysisPoints[i];
      wSum += p.composite_score;
      while (analysisPoints[wStart].time < p.time - smoothingWindowSec) {
        wSum -= analysisPoints[wStart].composite_score;
        wStart++;
      }
      rollingMeanByTime.set(p.time, wSum / (i - wStart + 1));
    }

    // --- Pass 2: downsample for display ---
    const flow: ControlChartDataPoint[] = [];
    const static_: ControlChartDataPoint[] = [];
    const composite: ControlChartDataPoint[] = [];
    const compositeSmoothed: ControlChartDataPoint[] = [];

    for (let i = 0; i < points.length; i += step) {
      const p = points[i];
      flow.push({ time: p.time, value: p.flow });

      if (p.phase === "analysis") {
        static_.push({ time: p.time, value: p.static_score });
        composite.push({ time: p.time, value: p.composite_score });
        const sm = rollingMeanByTime.get(p.time);
        compositeSmoothed.push({ time: p.time, value: sm ?? p.composite_score });
      }
    }

    return { flow, static_, composite, compositeSmoothed };
  }, [result, smoothingWindowSec]);

  // Find when traffic light goes red (approximate clogging time)
  const cloggingTime = useMemo(() => {
    if (!result) return null;
    const firstRed = result.timeseries.find((p) => p.traffic_light === "red");
    return firstRed ? firstRed.time : null;
  }, [result]);

  // Find threshold crossing times — updates with sigma and smoothing window.
  // Static: first raw sample above threshold (DC method, instantaneous is correct).
  // Composite: first time the rolling mean crosses the threshold, to suppress
  //            transient spikes and only flag a sustained spectral shift.
  const thresholdCrossings = useMemo(() => {
    if (!result) return { static: null as number | null, composite: null as number | null };

    const thresh = thresholds || {
      sigma: 3.0,
      fft_threshold: 0.05,
      static_threshold: 0.018,
    };

    const analysisPoints = result.timeseries.filter((p) => p.phase === "analysis");

    // --- Static: raw first crossing ---
    let staticCrossing: number | null = null;
    for (const p of analysisPoints) {
      if (p.static_score > thresh.static_threshold) {
        staticCrossing = p.time;
        break;
      }
    }

    // --- Composite: rolling mean first crossing ---
    let compositeCrossing: number | null = null;
    let wStart = 0;
    let wSum = 0;

    for (let i = 0; i < analysisPoints.length; i++) {
      const p = analysisPoints[i];
      wSum += p.composite_score;
      while (analysisPoints[wStart].time < p.time - smoothingWindowSec) {
        wSum -= analysisPoints[wStart].composite_score;
        wStart++;
      }
      const rollingMean = wSum / (i - wStart + 1);
      if (rollingMean > thresh.fft_threshold) {
        compositeCrossing = p.time;
        break;
      }
    }

    return { static: staticCrossing, composite: compositeCrossing };
  }, [result, thresholds, smoothingWindowSec]);

  const effectiveThresholds = thresholds || {
    sigma: 3.0,
    fft_threshold: 0.05,
    static_threshold: 0.018,
  };

  // ── ML analytics (analysis phase only) ────────────────────────────────────
  const trafficDist = useMemo(() => {
    if (!result) return null;
    const pts = result.timeseries.filter((p) => p.phase === "analysis");
    if (!pts.length) return null;
    const total = pts.length;
    return {
      green:  pts.filter((p) => p.traffic_light === "green").length  / total,
      yellow: pts.filter((p) => p.traffic_light === "yellow").length / total,
      red:    pts.filter((p) => p.traffic_light === "red").length    / total,
      total,
    };
  }, [result]);

  const mlStats = useMemo(() => {
    if (!result) return null;
    const pts = result.timeseries.filter(
      (p) => p.phase === "analysis" && p.ensemble_probability > 0,
    );
    if (!pts.length) return null;
    const peak = Math.max(...pts.map((p) => p.ensemble_probability));
    const firstCritical = pts.find((p) => p.ensemble_probability >= 0.7);
    const redPts = pts.filter((p) => p.traffic_light === "red");
    const avgRedProb =
      redPts.length > 0
        ? redPts.reduce((s, p) => s + p.ensemble_probability, 0) / redPts.length
        : null;
    return { peak, firstCriticalSec: firstCritical?.time ?? null, avgRedProb };
  }, [result]);

  const mlChartData = useMemo(() => {
    if (!result) return [];
    const pts = result.timeseries.filter((p) => p.phase === "analysis");
    const maxPts = 2000;
    const step = Math.max(1, Math.floor(pts.length / maxPts));
    return pts
      .filter((_, i) => i % step === 0)
      .map((p) => ({ time: p.time, ensemble: p.ensemble_probability }));
  }, [result]);

  return (
    <div>
      {/* Controls */}
      <div
        style={{
          padding: "20px",
          backgroundColor: "#fff",
          borderRadius: "8px",
          border: "1px solid #e5e7eb",
          marginBottom: "20px",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div>
            <h2 style={{ margin: 0, fontSize: "18px" }}>Batch Analysis</h2>
            <p
              style={{ margin: "4px 0 0", fontSize: "13px", color: "#6b7280" }}
            >
              Process entire file at once — no streaming, no timeouts
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <label style={{ fontSize: "13px", color: "#6b7280" }}>
                Calibration seconds:
              </label>
              <input
                type="number"
                value={calibrationSeconds}
                onChange={(e) =>
                  setCalibrationSeconds(Math.max(5, Number(e.target.value)))
                }
                min={5}
                max={300}
                step={5}
                style={{
                  width: "80px",
                  padding: "6px 8px",
                  borderRadius: "6px",
                  border: "1px solid #e5e7eb",
                  fontSize: "13px",
                }}
              />
            </div>
            <button
              onClick={runAnalysis}
              disabled={!selectedFile || loading}
              style={{
                padding: "10px 20px",
                borderRadius: "6px",
                border: "none",
                backgroundColor:
                  !selectedFile || loading ? "#d1d5db" : "#2563eb",
                color: "#fff",
                fontWeight: 600,
                cursor: !selectedFile || loading ? "not-allowed" : "pointer",
                fontSize: "14px",
              }}
            >
              {loading ? "Processing..." : "Run Analysis"}
            </button>
          </div>
        </div>

        {error && (
          <div
            style={{
              marginTop: "12px",
              padding: "10px",
              backgroundColor: "#fee2e2",
              color: "#991b1b",
              borderRadius: "6px",
              fontSize: "13px",
            }}
          >
            {error}
          </div>
        )}
      </div>

      {/* Loading indicator */}
      {loading && (
        <div
          style={{
            padding: "40px",
            textAlign: "center",
            backgroundColor: "#fff",
            borderRadius: "8px",
            border: "1px solid #e5e7eb",
            marginBottom: "20px",
          }}
        >
          <div
            style={{ fontSize: "16px", fontWeight: 500, marginBottom: "8px" }}
          >
            Processing file...
          </div>
          <div style={{ fontSize: "13px", color: "#6b7280" }}>
            This may take a moment for large datasets
          </div>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div>
          {/* Summary bar */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(5, 1fr)",
              gap: "12px",
              marginBottom: "20px",
            }}
          >
            <SummaryCard
              label="Total Points"
              value={result.metadata.total_points.toLocaleString()}
            />
            <SummaryCard
              label="Duration"
              value={`${(result.metadata.duration_seconds / 60).toFixed(1)} min`}
            />
            <SummaryCard
              label="Sampling"
              value={`${result.metadata.sampling_hz} Hz`}
            />
            <SummaryCard
              label="Clogging Detected"
              value={
                cloggingTime ? `${(cloggingTime / 60).toFixed(1)} min` : "None"
              }
              highlight={cloggingTime !== null}
            />
            <SummaryCard
              label="Calibration"
              value={`${result.calibration.samples_used} samples`}
            />
          </div>

          {/* Sigma selector */}
          <div
            style={{
              padding: "16px 20px",
              backgroundColor: "#fff",
              borderRadius: "8px",
              border: "1px solid #e5e7eb",
              marginBottom: "20px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <div>
              <span style={{ fontWeight: 600, fontSize: "14px" }}>
                Threshold σ:
              </span>
              <span
                style={{
                  fontSize: "12px",
                  color: "#6b7280",
                  marginLeft: "8px",
                }}
              >
                Composite: {effectiveThresholds.fft_threshold.toFixed(6)} |
                Static: {effectiveThresholds.static_threshold.toFixed(6)}
              </span>
            </div>
            <div style={{ display: "flex", gap: "6px" }}>
              {SIGMA_OPTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSigmaChange(s)}
                  style={{
                    padding: "6px 14px",
                    border: "2px solid",
                    borderColor: s === sigma ? "#7c3aed" : "#e5e7eb",
                    borderRadius: "6px",
                    background: s === sigma ? "#7c3aed" : "#fff",
                    color: s === sigma ? "#fff" : "#374151",
                    cursor: "pointer",
                    fontWeight: s === sigma ? 700 : 400,
                    fontSize: "13px",
                    transition: "all 0.15s ease",
                  }}
                >
                  {s}σ
                </button>
              ))}
            </div>
          </div>

          {/* Threshold crossing times — updates instantly when sigma / smoothing changes */}
          <div
            style={{
              padding: "16px 20px",
              backgroundColor: "#fff",
              borderRadius: "8px",
              border: "1px solid #e5e7eb",
              marginBottom: "20px",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: "12px",
              }}
            >
              <span style={{ fontSize: "13px", fontWeight: 600, color: "#374151" }}>
                Sigma Threshold Crossings ({sigma}σ)
              </span>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <label style={{ fontSize: "12px", color: "#6b7280" }}>
                  Composite smoothing window:
                </label>
                <input
                  type="number"
                  value={smoothingWindowSec}
                  onChange={(e) =>
                    setSmoothingWindowSec(Math.max(5, Number(e.target.value)))
                  }
                  min={5}
                  max={600}
                  step={5}
                  style={{
                    width: "70px",
                    padding: "4px 6px",
                    borderRadius: "6px",
                    border: "1px solid #e5e7eb",
                    fontSize: "12px",
                  }}
                />
                <span style={{ fontSize: "12px", color: "#6b7280" }}>s</span>
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
              <CrossingCard
                label="Static — first raw sample above threshold"
                time={thresholdCrossings.static}
                threshold={effectiveThresholds.static_threshold}
                color="#2E7D32"
              />
              <CrossingCard
                label={`Composite — rolling mean (${smoothingWindowSec}s) crosses threshold`}
                time={thresholdCrossings.composite}
                threshold={effectiveThresholds.fft_threshold}
                color="#800080"
              />
            </div>
          </div>

          {/* Calibration details (collapsible) */}
          <details
            style={{
              marginBottom: "20px",
              padding: "16px 20px",
              backgroundColor: "#f9fafb",
              borderRadius: "8px",
              border: "1px solid #e5e7eb",
            }}
          >
            <summary
              style={{ cursor: "pointer", fontWeight: 500, fontSize: "14px" }}
            >
              Calibration Details
            </summary>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(4, 1fr)",
                gap: "12px",
                marginTop: "12px",
                fontSize: "13px",
              }}
            >
              <div>
                <div style={{ color: "#6b7280" }}>Baseline Mean (dP)</div>
                <div style={{ fontWeight: 600 }}>
                  {result.calibration.baseline_mean.toFixed(6)}
                </div>
              </div>
              <div>
                <div style={{ color: "#6b7280" }}>Baseline Std (dP)</div>
                <div style={{ fontWeight: 600 }}>
                  {result.calibration.baseline_std.toFixed(6)}
                </div>
              </div>
              <div>
                <div style={{ color: "#6b7280" }}>Composite Mean</div>
                <div style={{ fontWeight: 600 }}>
                  {result.calibration.composite_mean.toFixed(6)}
                </div>
              </div>
              <div>
                <div style={{ color: "#6b7280" }}>Composite Std</div>
                <div style={{ fontWeight: 600 }}>
                  {result.calibration.composite_std.toFixed(6)}
                </div>
              </div>
            </div>
          </details>

          {/* Charts */}
          <div
            style={{ display: "flex", flexDirection: "column", gap: "16px" }}
          >
            <ControlChart
              title={`Method: STATIC (σ=${sigma})`}
              data={displayData.static_}
              threshold={effectiveThresholds.static_threshold}
              color="#2E7D32"
              useLogScale={false}
              height={220}
            />
            <ControlChart
              title={`Method: COMPOSITE (σ=${sigma})`}
              data={displayData.composite}
              threshold={effectiveThresholds.fft_threshold}
              color="#800080"
              useLogScale={true}
              height={220}
              overlayData={displayData.compositeSmoothed}
              overlayColor="#f59e0b"
              overlayLabel={`Rolling mean (${smoothingWindowSec}s)`}
            />
            <ControlChart
              title="Flow Rate"
              data={displayData.flow}
              threshold={Infinity}
              color="#0000FF"
              unit="Kg/h"
              useLogScale={false}
              height={200}
            />
          </div>

          {/* ── ML Results Section ── */}
          {(trafficDist || mlStats) && (
            <BatchMLSection
              trafficDist={trafficDist}
              mlStats={mlStats}
              mlChartData={mlChartData}
            />
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// BatchMLSection — ML probability analytics added after the physics charts
// ---------------------------------------------------------------------------

interface TrafficDist {
  green: number;
  yellow: number;
  red: number;
  total: number;
}

interface MLStats {
  peak: number;
  firstCriticalSec: number | null;
  avgRedProb: number | null;
}

interface MLChartPoint {
  time: number;
  ensemble: number;
}

function BatchMLSection({
  trafficDist,
  mlStats,
  mlChartData,
}: {
  trafficDist: TrafficDist | null;
  mlStats: MLStats | null;
  mlChartData: MLChartPoint[];
}) {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  return (
    <div style={{ marginTop: "20px" }}>
      {/* ── Header ── */}
      <div
        style={{
          padding: "16px 20px",
          backgroundColor: "#fff",
          borderRadius: "8px",
          border: "1px solid #e5e7eb",
          marginBottom: "12px",
        }}
      >
        <h3 style={{ margin: "0 0 16px", fontSize: "16px", fontWeight: 700 }}>
          ML Detection Results
        </h3>

        {/* Traffic light distribution bar */}
        {trafficDist && (
          <div style={{ marginBottom: "16px" }}>
            <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "6px", fontWeight: 500 }}>
              Signal State Distribution ({trafficDist.total.toLocaleString()} analysis samples)
            </div>
            <div
              style={{
                display: "flex",
                height: "28px",
                borderRadius: "6px",
                overflow: "hidden",
                border: "1px solid #e5e7eb",
              }}
            >
              {trafficDist.green > 0.005 && (
                <div
                  title={`Healthy: ${pct(trafficDist.green)}`}
                  style={{ flex: trafficDist.green, backgroundColor: "#22c55e" }}
                />
              )}
              {trafficDist.yellow > 0.005 && (
                <div
                  title={`Warning: ${pct(trafficDist.yellow)}`}
                  style={{ flex: trafficDist.yellow, backgroundColor: "#f59e0b" }}
                />
              )}
              {trafficDist.red > 0.005 && (
                <div
                  title={`Clogged: ${pct(trafficDist.red)}`}
                  style={{ flex: trafficDist.red, backgroundColor: "#ef4444" }}
                />
              )}
            </div>
            <div style={{ display: "flex", gap: "20px", marginTop: "6px", fontSize: "12px" }}>
              <span style={{ color: "#16a34a" }}>
                <strong>{pct(trafficDist.green)}</strong> Healthy
              </span>
              <span style={{ color: "#d97706" }}>
                <strong>{pct(trafficDist.yellow)}</strong> Warning
              </span>
              <span style={{ color: "#dc2626" }}>
                <strong>{pct(trafficDist.red)}</strong> Clogged
              </span>
            </div>
          </div>
        )}

        {/* ML stats cards */}
        {mlStats && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, 1fr)",
              gap: "12px",
            }}
          >
            <MLStatCard
              label="Peak ML Probability"
              value={pct(mlStats.peak)}
              highlight={mlStats.peak >= 0.7}
            />
            <MLStatCard
              label="First Critical Alert"
              value={
                mlStats.firstCriticalSec !== null
                  ? formatHHMMSS(mlStats.firstCriticalSec)
                  : "None"
              }
              highlight={mlStats.firstCriticalSec !== null}
            />
            <MLStatCard
              label="Avg Probability (clogged)"
              value={mlStats.avgRedProb !== null ? pct(mlStats.avgRedProb) : "—"}
              highlight={(mlStats.avgRedProb ?? 0) >= 0.7}
            />
          </div>
        )}
      </div>

      {/* ── Ensemble probability chart ── */}
      {mlChartData.length > 0 && (
        <div
          style={{
            padding: "16px 20px",
            backgroundColor: "#fff",
            borderRadius: "8px",
            border: "1px solid #e5e7eb",
          }}
        >
          <div style={{ fontSize: "14px", fontWeight: 600, color: "#374151", marginBottom: "12px" }}>
            Ensemble ML Probability Over Time
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={mlChartData} margin={{ top: 8, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

              {/* Zone bands */}
              <ReferenceArea y1={0}   y2={0.2} fill="#22c55e" fillOpacity={0.07} ifOverflow="hidden" />
              <ReferenceArea y1={0.2} y2={0.7} fill="#f59e0b" fillOpacity={0.07} ifOverflow="hidden" />
              <ReferenceArea y1={0.7} y2={1}   fill="#ef4444" fillOpacity={0.07} ifOverflow="hidden" />

              <XAxis
                dataKey="time"
                tickFormatter={(v) => `${Number(v).toFixed(0)}s`}
                stroke="#9ca3af"
                fontSize={11}
              />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                stroke="#9ca3af"
                fontSize={11}
              />
              <Tooltip
                formatter={(v) => [`${(Number(v) * 100).toFixed(1)}%`, "Ensemble"]}
                labelFormatter={(l) => `Time: ${Number(l).toFixed(1)}s`}
              />
              <Area
                type="monotone"
                dataKey="ensemble"
                name="Ensemble"
                stroke="#8b5cf6"
                strokeWidth={2}
                fill="#8b5cf6"
                fillOpacity={0.15}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function MLStatCard({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div
      style={{
        padding: "12px 14px",
        backgroundColor: highlight ? "#fef2f2" : "#f9fafb",
        borderRadius: "8px",
        border: `1px solid ${highlight ? "#fca5a5" : "#e5e7eb"}`,
      }}
    >
      <div style={{ fontSize: "11px", color: "#6b7280", marginBottom: "4px" }}>{label}</div>
      <div
        style={{
          fontSize: "18px",
          fontWeight: 700,
          color: highlight ? "#dc2626" : "#111827",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function formatHHMMSS(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
}

function CrossingCard({
  label,
  time,
  threshold,
  color,
}: {
  label: string;
  time: number | null;
  threshold: number;
  color: string;
}) {
  const hasCrossing = time !== null;
  return (
    <div
      style={{
        padding: "14px 16px",
        backgroundColor: hasCrossing ? "#fef2f2" : "#f9fafb",
        borderRadius: "8px",
        border: `1px solid ${hasCrossing ? "#fca5a5" : "#e5e7eb"}`,
      }}
    >
      <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "6px" }}>
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: "10px" }}>
        <span
          style={{
            fontSize: "20px",
            fontWeight: 700,
            color: hasCrossing ? "#dc2626" : "#9ca3af",
            fontVariantNumeric: "tabular-nums",
            letterSpacing: "0.02em",
          }}
        >
          {hasCrossing ? formatHHMMSS(time!) : "No crossing"}
        </span>
      </div>
      <div style={{ marginTop: "4px", fontSize: "11px", color }}>
        Threshold: {threshold.toFixed(6)}
      </div>
    </div>
  );
}

function SummaryCard({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div
      style={{
        padding: "14px",
        backgroundColor: highlight ? "#fef2f2" : "#fff",
        borderRadius: "8px",
        border: `1px solid ${highlight ? "#fca5a5" : "#e5e7eb"}`,
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>
        {label}
      </div>
      <div
        style={{
          fontSize: "16px",
          fontWeight: 700,
          color: highlight ? "#090303" : "#111827",
        }}
      >
        {value}
      </div>
    </div>
  );
}
