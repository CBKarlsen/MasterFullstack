/**
 * ControlChartGrid — lays out a stack of ControlChart panels for the live view.
 *
 * Renders two sections: (A) raw sensor columns (e.g. live flow rate) selected by
 * the user, and (B) anomaly-detection score series (STATIC and COMPOSITE),
 * each drawn as a control chart with its threshold/control-limit line. Consumes
 * `rawData` (raw sensor samples) and `chartData` (per-sample detection scores),
 * both downsampled to the last `maxDataPoints` points. Thresholds are resolved
 * with a precedence of backend-supplied value > explicit prop > built-in default;
 * sensor columns with no supplied threshold fall back to a dynamic mean+kσ limit
 * computed locally. Also exposes the sigma-selector buttons that let the user
 * change the global detection threshold via `onSigmaChange`.
 */
import { useMemo } from "react";
import type { ChartDataPoint, RawDataPoint } from "../types";
import { ControlChart, type ControlChartDataPoint } from "./ControlChart";

interface MethodThresholds {
	composite?: number;
	static?: number;
	slope?: number;
	ensemble?: number;
}

interface ControlChartGridProps {
	rawData: RawDataPoint[];
	selectedColumns: string[];
	chartData: ChartDataPoint[];
	methodThresholds?: MethodThresholds;
	defaultColumnThreshold?: number;
	columnThresholds?: Record<string, number>;
	maxDataPoints?: number;
	chartHeight?: number;
	/** Current sigma value from backend */
	currentSigma?: number;
	/** Callback when user changes sigma */
	onSigmaChange?: (sigma: number) => void;
}

// --- METHOD CONFIGURATION ---
// Keys must match what simulationStore puts into chartData:
//   point.static, point.composite, point.slope
const METHODS = [
	{
		key: "static", // matches store: point.static
		label: "Method: STATIC",
		useLog: false,
		defaultThreshold: 0.018,
		color: "#2E7D32", // Dark green
	},
	{
		key: "composite", // matches store: point.composite
		label: "Method: COMPOSITE",
		useLog: true, // Log scale
		defaultThreshold: 0.05,
		color: "#800080", // Purple
	},
] as const;

const SIGMA_OPTIONS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

const COLUMN_COLORS = ["#0000FF", "#0288d1", "#0097a7"];

// Infer the display unit for a raw sensor column from its name.
const getColumnUnit = (column: string): string => {
	const lower = column.toLowerCase();
	if (lower.includes("flow")) return "L/s";
	if (lower.includes("pressure")) return "bar";
	return "";
};

// Fallback control limit for sensor columns: mean + multiplier·stddev of the
// supplied values (a kσ upper control limit), used when no threshold is given.
function calculateDynamicThreshold(
	values: number[],
	multiplier: number = 5,
): number {
	if (values.length === 0) return 1;
	const validValues = values.filter((v) => !isNaN(v) && isFinite(v));
	if (validValues.length === 0) return 1;
	const mean = validValues.reduce((a, b) => a + b, 0) / validValues.length;
	const squaredDiffs = validValues.map((v) => (v - mean) ** 2);
	const variance = squaredDiffs.reduce((a, b) => a + b, 0) / validValues.length;
	return mean + multiplier * Math.sqrt(variance);
}

export function ControlChartGrid({
	rawData,
	selectedColumns,
	chartData,
	methodThresholds = {},
	defaultColumnThreshold,
	columnThresholds = {},
	maxDataPoints = 300,
	chartHeight = 250,
	currentSigma = 3.0,
	onSigmaChange,
}: ControlChartGridProps) {
	const limitedRawData = useMemo(() => {
		return rawData.length > maxDataPoints
			? rawData.slice(-maxDataPoints)
			: rawData;
	}, [rawData, maxDataPoints]);

	const limitedChartData = useMemo(() => {
		return chartData.length > maxDataPoints
			? chartData.slice(-maxDataPoints)
			: chartData;
	}, [chartData, maxDataPoints]);

	// Column data (Live Flow Rate)
	const columnChartData = useMemo(() => {
		const result: Record<string, ControlChartDataPoint[]> = {};
		selectedColumns.forEach((col) => {
			result[col] = limitedRawData
				.filter((point) => point[col] !== undefined && point[col] !== null)
				.map((point) => ({
					time: point.time,
					value: point[col] as number,
				}));
		});
		return result;
	}, [limitedRawData, selectedColumns]);

	// Method data (STATIC + COMPOSITE)
	const methodChartData = useMemo(() => {
		const result: Record<string, ControlChartDataPoint[]> = {};
		METHODS.forEach(({ key }) => {
			result[key] = limitedChartData
				.filter((point) => point[key] !== undefined && point[key] !== null)
				.map((point) => ({
					time: point.time,
					value: point[key] as number,
				}));
		});
		return result;
	}, [limitedChartData]);

	// Column thresholds
	const calculatedColumnThresholds = useMemo(() => {
		const result: Record<string, number> = {};
		selectedColumns.forEach((col) => {
			if (columnThresholds[col] !== undefined) {
				result[col] = columnThresholds[col];
			} else if (defaultColumnThreshold !== undefined) {
				result[col] = defaultColumnThreshold;
			} else {
				const values = columnChartData[col]?.map((d) => d.value) || [];
				result[col] = calculateDynamicThreshold(values, 5);
			}
		});
		return result;
	}, [
		selectedColumns,
		columnChartData,
		columnThresholds,
		defaultColumnThreshold,
	]);

	// Extract the most recent per-method thresholds and active sigma from the
	// last chartData frame (the backend embeds these on every sample); falls
	// back to defaults when no data has arrived yet.
	const latestThresholds = useMemo(() => {
		if (limitedChartData.length === 0)
			return { static: 0.018, composite: 0.05, sigma: 3.0 };
		const latest = limitedChartData[limitedChartData.length - 1] as {
			static_threshold?: number;
			limit_threshold?: number;
			current_sigma?: number;
		};
		return {
			static:
				latest.static_threshold && latest.static_threshold > 0
					? latest.static_threshold
					: 0.018,
			composite:
				latest.limit_threshold && latest.limit_threshold > 0
					? latest.limit_threshold
					: 0.05,
			sigma: latest.current_sigma ?? 3.0,
		};
	}, [limitedChartData]);

	const effectiveSigma = latestThresholds.sigma ?? currentSigma;

	return (
		<div>
			{/* SECTION A: SENSOR DATA */}
			{selectedColumns.length > 0 && (
				<div style={{ marginBottom: "24px" }}>
					<h2 style={{ margin: "0 0 16px", fontSize: "18px", fontWeight: 500 }}>
						Live Flow Rate
					</h2>
					<div
						style={{ display: "grid", gridTemplateColumns: "1fr", gap: "16px" }}
					>
						{selectedColumns.map((col, index) => {
							const data = columnChartData[col];
							if (!data || data.length === 0) return null;
							return (
								<ControlChart
									key={col}
									title={col}
									data={data}
									threshold={calculatedColumnThresholds[col]}
									color={COLUMN_COLORS[index % COLUMN_COLORS.length]}
									unit={getColumnUnit(col)}
									useLogScale={false}
									height={chartHeight}
								/>
							);
						})}
					</div>
				</div>
			)}

			{/* SECTION B: ANOMALY DETECTION */}
			<div>
				<div
					style={{
						display: "flex",
						justifyContent: "space-between",
						alignItems: "center",
						margin: "0 0 16px",
					}}
				>
					<h2 style={{ margin: 0, fontSize: "18px", fontWeight: 500 }}>
						Anomaly Detection Analysis
					</h2>

					{onSigmaChange && (
						<div
							style={{
								display: "flex",
								alignItems: "center",
								gap: "8px",
								fontSize: "13px",
							}}
						>
							<span style={{ fontWeight: 500 }}>Threshold σ:</span>
							{SIGMA_OPTIONS.map((s) => (
								<button
									key={s}
									onClick={() => onSigmaChange(s)}
									style={{
										padding: "4px 10px",
										border: "1px solid",
										borderColor: s === effectiveSigma ? "#800080" : "#ccc",
										borderRadius: "4px",
										background: s === effectiveSigma ? "#800080" : "#fff",
										color: s === effectiveSigma ? "#fff" : "#333",
										cursor: "pointer",
										fontWeight: s === effectiveSigma ? 600 : 400,
										fontSize: "12px",
										transition: "all 0.15s ease",
									}}
								>
									{s}σ
								</button>
							))}
						</div>
					)}
				</div>

				<div
					style={{ display: "grid", gridTemplateColumns: "1fr", gap: "16px" }}
				>
					{METHODS.map(({ key, label, useLog, defaultThreshold, color }) => {
						const data = methodChartData[key] || [];

						// Resolve threshold: backend > props > default
						let threshold: number;
						if (key === "static") {
							threshold =
								methodThresholds.static ??
								latestThresholds.static ??
								defaultThreshold;
						} else if (key === "composite") {
							threshold =
								methodThresholds.composite ??
								latestThresholds.composite ??
								defaultThreshold;
						} else {
							threshold = defaultThreshold;
						}

						return (
							<ControlChart
								key={key}
								title={`${label} (σ=${effectiveSigma})`}
								data={data}
								threshold={threshold}
								color={color}
								useLogScale={useLog}
								height={chartHeight}
							/>
						);
					})}
				</div>
			</div>
		</div>
	);
}
