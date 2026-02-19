import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

export interface ControlChartDataPoint {
  time: number;
  value: number;
}

interface ControlChartProps {
  title: string;
  data: ControlChartDataPoint[];
  threshold: number;
  color?: string;
  unit?: string;
  useLogScale?: boolean;
  height?: number;
}

const COLORS = {
  normal: "#2196F3",
  breach: "#F44336",
  grid: "#e5e7eb",
  axis: "#9ca3af",
};

const LOG_SAFE_MIN = 0.0001;

// Format time for display (e.g., 1234s -> "1234s" or "20:34" for longer times)
const formatTime = (seconds: number): string => {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m${secs.toString().padStart(2, "0")}s`;
  }
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h${mins.toString().padStart(2, "0")}m`;
};

export function ControlChart({
  title,
  data,
  threshold,
  color = COLORS.normal,
  unit = "",
  useLogScale = false,
  height = 200,
}: ControlChartProps) {
  const { chartData, isBreaching, yDomain, xDomain } = useMemo(() => {
    if (data.length === 0) {
      return {
        chartData: [],
        isBreaching: false,
        yDomain: ["auto", "auto"] as [string, string],
        xDomain: [0, 100] as [number, number],
      };
    }

    const latestValue = data[data.length - 1].value;
    const isBreaching = latestValue > threshold;

    // Transform data for display - use absolute time
    const chartData = data.map((point) => ({
      ...point,
      safeValue: useLogScale && point.value <= 0 ? LOG_SAFE_MIN : point.value,
    }));

    // Calculate X-Axis Domain (dynamic, based on actual time range)
    const minTime = data[0].time;
    const maxTime = data[data.length - 1].time;
    const xDomain: [number, number] = [minTime, maxTime];

    // Calculate Y-Axis Domain
    let minVal = Math.min(...chartData.map((d) => d.safeValue));
    let maxVal = Math.max(...chartData.map((d) => d.safeValue));

    // Ensure domain includes the threshold line
    if (isFinite(threshold)) {
      maxVal = Math.max(maxVal, threshold);
    }

    let yDomain: [number | string, number | string] = ["auto", "auto"];

    if (useLogScale) {
      yDomain = [Math.max(minVal * 0.5, LOG_SAFE_MIN), maxVal * 2];
    } else {
      yDomain = [0, maxVal * 1.2];
    }

    return { chartData, isBreaching, yDomain, xDomain };
  }, [data, threshold, useLogScale]);

  // Formatters
  const formatXAxis = (val: number) => formatTime(val);
  const formatYAxis = (val: number) => {
    if (Math.abs(val) < 0.01) return val.toExponential(1);
    return val.toFixed(2);
  };

  if (data.length === 0)
    return (
      <div
        style={{
          height,
          padding: "16px",
          backgroundColor: "#fff",
          borderRadius: "12px",
          border: "1px solid #e5e7eb",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#9ca3af",
        }}
      >
        <div style={{ textAlign: "center" }}>
          <div
            style={{ fontSize: "14px", fontWeight: 500, marginBottom: "4px" }}
          >
            {title}
          </div>
          <div style={{ fontSize: "12px" }}>
            Waiting for calibration to complete...
          </div>
        </div>
      </div>
    );

  const currentValue = data[data.length - 1].value;

  return (
    <div
      style={{
        padding: "16px",
        backgroundColor: "#fff",
        borderRadius: "12px",
        border: isBreaching ? "2px solid #F44336" : "1px solid #e5e7eb",
        boxShadow: "0 2px 4px rgba(0,0,0,0.05)",
        transition: "border-color 0.3s ease",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: "10px",
        }}
      >
        <div>
          <h3 style={{ margin: 0, fontSize: "14px", color: "#555" }}>
            {title}
          </h3>
          <span style={{ fontSize: "11px", color: "#999" }}>
            Limit: {threshold.toFixed(4)}
          </span>
        </div>
        <div style={{ textAlign: "right" }}>
          <div
            style={{
              fontSize: "18px",
              fontWeight: "bold",
              color: isBreaching ? COLORS.breach : "#333",
            }}
          >
            {currentValue.toFixed(4)}{" "}
            <span style={{ fontSize: "12px" }}>{unit}</span>
          </div>
          {isBreaching && (
            <div
              style={{
                color: COLORS.breach,
                fontSize: "10px",
                fontWeight: "bold",
              }}
            >
              LIMIT EXCEEDED
            </div>
          )}
        </div>
      </div>

      {/* Chart */}
      <div style={{ width: "100%", height: height, minWidth: 200 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <defs>
              <linearGradient
                id={`grad-${title.replace(/\s/g, "-")}`}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop
                  offset="5%"
                  stopColor={isBreaching ? COLORS.breach : color}
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor={isBreaching ? COLORS.breach : color}
                  stopOpacity={0.0}
                />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
              stroke={COLORS.grid}
            />

            {/* X-Axis: Dynamic domain showing absolute time */}
            <XAxis
              dataKey="time"
              domain={xDomain}
              type="number"
              tickFormatter={formatXAxis}
              stroke={COLORS.axis}
              fontSize={10}
            />

            <YAxis
              domain={yDomain as [number | string, number | string]}
              scale={useLogScale ? "log" : "auto"}
              tickFormatter={formatYAxis}
              stroke={COLORS.axis}
              fontSize={10}
              width={50}
            />

            <Tooltip
              labelFormatter={(v) => `Time: ${formatTime(Number(v))}`}
              formatter={(v: number) => [v.toFixed(5), title]}
            />

            {/* The Fixed Backend Threshold */}
            {isFinite(threshold) && (
              <ReferenceLine
                y={threshold}
                stroke="#FF8C00"
                strokeDasharray="6 3"
                label={{
                  value: "LIMIT",
                  position: "right",
                  fill: "#FF8C00",
                  fontSize: 10,
                }}
              />
            )}

            <Area
              type="monotone"
              dataKey="safeValue"
              stroke={isBreaching ? COLORS.breach : color}
              fill={`url(#grad-${title.replace(/\s/g, "-")})`}
              strokeWidth={2}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
