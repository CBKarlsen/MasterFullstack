import { useState, useMemo, useCallback } from "react";
import axios from "axios";
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
  const [calibrationSamples, setCalibrationSamples] = useState(400);

  // Run full analysis
  const runAnalysis = useCallback(async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post<AnalysisResult>(
        `/api/analyze?file=${encodeURIComponent(selectedFile)}&sigma=${sigma}&calibration_samples=${calibrationSamples}`,
      );
      setResult(response.data);
      setThresholds(response.data.thresholds);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  }, [selectedFile, sigma]);

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
  const displayData = useMemo(() => {
    if (!result) return { flow: [], static_: [], composite: [] };

    const points = result.timeseries;
    const maxDisplay = 3000;
    const step = Math.max(1, Math.floor(points.length / maxDisplay));

    const flow: ControlChartDataPoint[] = [];
    const static_: ControlChartDataPoint[] = [];
    const composite: ControlChartDataPoint[] = [];

    for (let i = 0; i < points.length; i += step) {
      const p = points[i];
      flow.push({ time: p.time, value: p.flow });

      // Only include analysis-phase data for detection charts
      if (p.phase === "analysis") {
        static_.push({ time: p.time, value: p.static_score });
        composite.push({ time: p.time, value: p.composite_score });
      }
    }

    return { flow, static_, composite };
  }, [result]);

  // Find when traffic light goes red (approximate clogging time)
  const cloggingTime = useMemo(() => {
    if (!result) return null;
    const firstRed = result.timeseries.find((p) => p.traffic_light === "red");
    return firstRed ? firstRed.time : null;
  }, [result]);

  const effectiveThresholds = thresholds || {
    sigma: 3.0,
    fft_threshold: 0.05,
    static_threshold: 0.018,
  };

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
                Calibration samples:
              </label>
              <input
                type="number"
                value={calibrationSamples}
                onChange={(e) =>
                  setCalibrationSamples(Math.max(50, Number(e.target.value)))
                }
                min={50}
                max={2000}
                step={50}
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
              title="Flow Rate"
              data={displayData.flow}
              threshold={Infinity}
              color="#0000FF"
              unit="L/s"
              useLogScale={false}
              height={200}
            />
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
            />
          </div>
        </div>
      )}
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
