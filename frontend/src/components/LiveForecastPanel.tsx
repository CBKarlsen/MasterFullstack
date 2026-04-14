import { useMemo, useState } from "react";
import type { ChartDataPoint } from "../types";
import { computeForecast } from "../utils/forecastClient";
import { CloggingForecast } from "./CloggingForecast";
import type { AnalysisPoint } from "./BatchAnalysisCards";

interface LiveForecastPanelProps {
  chartData: ChartDataPoint[];
  fftThreshold: number | undefined;
}

// Map live ChartDataPoint format to the shape computeForecast expects.
// Calibration points have no `composite` value — filtering them out is
// equivalent to the `phase === "analysis"` filter used in batch mode.
function adaptToAnalysisPoints(chartData: ChartDataPoint[]): AnalysisPoint[] {
  return chartData
    .filter((p) => typeof p.composite === "number")
    .map((p) => ({
      time: p.time,
      composite_score: p.composite as number,
      phase: "analysis",
      // Fields not used by computeForecast — provide typed defaults
      flow: 0,
      pressure_drop: 0,
      static_score: 0,
      turbulence_score: 0,
      spectral_slope: 0,
      wavelet_score: 0,
      traffic_light: "",
      light_msg: "",
      ensemble_probability: 0,
    }));
}

export function LiveForecastPanel({ chartData, fftThreshold }: LiveForecastPanelProps) {
  const [criticalMultiplier, setCriticalMultiplier] = useState(2.0);
  const adapted = useMemo(() => adaptToAnalysisPoints(chartData), [chartData]);

  const forecast = useMemo(
    () =>
      fftThreshold !== undefined
        ? computeForecast(adapted, fftThreshold, { criticalMultiplier })
        : null,
    [adapted, fftThreshold, criticalMultiplier],
  );

  if (!forecast) return null;

  return (
    <div style={{ marginTop: "20px" }}>
      <div style={{
        fontSize: "12px", color: "#6b7280", marginBottom: "8px",
        display: "flex", alignItems: "center", gap: "6px",
      }}>
        <span style={{
          width: "7px", height: "7px", borderRadius: "50%",
          backgroundColor: "#22c55e", display: "inline-block",
          boxShadow: "0 0 0 2px #dcfce7",
        }} />
        Live forecast — refits automatically as new data arrives
      </div>
      <CloggingForecast
        forecast={forecast}
        timeseries={adapted}
        criticalMultiplier={criticalMultiplier}
        onCriticalMultiplierChange={setCriticalMultiplier}
      />
    </div>
  );
}
