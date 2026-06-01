/**
 * CloggingForecast — projects the composite score's future trajectory once an
 * onset has been detected.
 *
 * Overlays the actual post-onset composite score against three fitted growth
 * models (linear, exponential, power law) and marks two horizontal reference
 * lines: the detection threshold and the critical level (a user-selected
 * multiple of that threshold). For each model it shows an R² goodness-of-fit
 * and an ETA to the critical crossing, highlighting the backend's chosen best
 * fit and a consensus ETA (median across models). Consumes a precomputed
 * `ForecastData` object (model fits, fitted curve points, thresholds, onset
 * time) plus the raw `timeseries`; the multiplier dropdown re-requests the
 * forecast via `onCriticalMultiplierChange`. The "Actual" series is downsampled
 * to at most ACTUAL_DOWNSAMPLE_MAX points to keep the chart responsive.
 */
import { useMemo } from "react";
import {
	CartesianGrid,
	ComposedChart,
	Legend,
	Line,
	ReferenceLine,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";
import type { FitResult, ForecastData } from "../utils/forecastClient";
import type { AnalysisPoint } from "./BatchAnalysisCards";

export type { ForecastData };

const MULTIPLIER_OPTIONS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0] as const;

interface CloggingForecastProps {
	forecast: ForecastData;
	timeseries: AnalysisPoint[];
	criticalMultiplier: number;
	onCriticalMultiplierChange: (v: number) => void;
}

// ── Constants ─────────────────────────────────────────────────────────────

const MODEL_COLORS: Record<string, string> = {
	linear: "#3b82f6",
	exponential: "#f97316",
	power_law: "#ef4444",
};

const MODEL_LABELS: Record<string, string> = {
	linear: "Linear",
	exponential: "Exponential",
	power_law: "Power Law",
};

const ACTUAL_DOWNSAMPLE_MAX = 300;

// ── Helpers ────────────────────────────────────────────────────────────────

// Format a duration in seconds as zero-padded HH:MM:SS.
function formatHHMMSS(seconds: number): string {
	const h = Math.floor(seconds / 3600);
	const m = Math.floor((seconds % 3600) / 60);
	const s = Math.floor(seconds % 60);
	return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
}

// Merge the downsampled actual composite series and the fitted model curves
// into a single row-per-timestamp dataset that Recharts can render.
function buildChartData(
	timeseries: AnalysisPoint[],
	onsetTime: number,
	curveData: Record<string, Array<{ time: number; value: number }>>,
): Record<string, number | undefined>[] {
	// Keep only post-onset analysis-phase samples — the forecast region.
	const postOnset = timeseries.filter(
		(p) => p.phase === "analysis" && p.time >= onsetTime,
	);
	// Downsample by taking every `step`-th point to cap at ACTUAL_DOWNSAMPLE_MAX.
	const step = Math.max(
		1,
		Math.floor(postOnset.length / ACTUAL_DOWNSAMPLE_MAX),
	);
	const sampled = postOnset.filter((_, i) => i % step === 0);

	const actualMap = new Map(sampled.map((p) => [p.time, p.composite_score]));
	const curveMaps = Object.fromEntries(
		Object.entries(curveData).map(([name, pts]) => [
			name,
			new Map(pts.map((p) => [p.time, p.value])),
		]),
	);
	// Union of all timestamps across the actual series and every model curve,
	// so each rendered row can carry whichever series have a value at that time.
	const allTimes = Array.from(
		new Set([
			...sampled.map((p) => p.time),
			...Object.values(curveData).flatMap((pts) => pts.map((p) => p.time)),
		]),
	).sort((a, b) => a - b);

	return allTimes.map((t) => ({
		time: t,
		actual: actualMap.get(t),
		...Object.fromEntries(
			Object.entries(curveMaps).map(([n, m]) => [n, m.get(t)]),
		),
	}));
}

// ── Subcomponents ──────────────────────────────────────────────────────────

interface ModelFitCardProps {
	name: string;
	fit: FitResult;
	isBest: boolean;
	criticalMultiplier: number;
}

// Summary card for one fitted growth model: color swatch, R² (green/amber/red
// by quality), and ETA to the critical threshold (or "No crossing").
function ModelFitCard({
	name,
	fit,
	isBest,
	criticalMultiplier,
}: ModelFitCardProps) {
	const color = MODEL_COLORS[name] ?? "#6b7280";
	const hasEta = fit.eta_seconds !== null;
	return (
		<div
			style={{
				padding: "12px 14px",
				borderRadius: "8px",
				border: `2px solid ${isBest ? color : "#e5e7eb"}`,
				backgroundColor: isBest ? `${color}10` : "#f9fafb",
			}}
		>
			<div
				style={{
					display: "flex",
					alignItems: "center",
					gap: "8px",
					marginBottom: "8px",
				}}
			>
				<span
					style={{
						width: "10px",
						height: "10px",
						borderRadius: "50%",
						backgroundColor: color,
						flexShrink: 0,
					}}
				/>
				<span style={{ fontSize: "13px", fontWeight: 700, color: "#374151" }}>
					{MODEL_LABELS[name] ?? name}
				</span>
				{isBest && (
					<span
						style={{
							fontSize: "10px",
							fontWeight: 700,
							marginLeft: "auto",
							padding: "2px 6px",
							borderRadius: "10px",
							backgroundColor: `${color}20`,
							color,
						}}
					>
						Best fit
					</span>
				)}
			</div>
			<div
				style={{
					fontSize: "12px",
					display: "flex",
					flexDirection: "column",
					gap: "4px",
				}}
			>
				<div style={{ display: "flex", justifyContent: "space-between" }}>
					<span style={{ color: "#6b7280" }}>R²</span>
					<span
						style={{
							fontWeight: 700,
							color:
								fit.r2 >= 0.8
									? "#16a34a"
									: fit.r2 >= 0.5
										? "#d97706"
										: "#dc2626",
						}}
					>
						{fit.r2.toFixed(3)}
					</span>
				</div>
				<div style={{ display: "flex", justifyContent: "space-between" }}>
					<span style={{ color: "#6b7280" }}>
						ETA to {criticalMultiplier}× threshold
					</span>
					<span
						style={{ fontWeight: 700, color: hasEta ? "#dc2626" : "#9ca3af" }}
					>
						{fit.eta_seconds !== null
							? formatHHMMSS(fit.eta_seconds)
							: "No crossing"}
					</span>
				</div>
			</div>
		</div>
	);
}

// ── Main component ─────────────────────────────────────────────────────────

export function CloggingForecast({
	forecast,
	timeseries,
	criticalMultiplier,
	onCriticalMultiplierChange,
}: CloggingForecastProps) {
	const chartData = useMemo(
		() => buildChartData(timeseries, forecast.onset_time, forecast.curve_data),
		[timeseries, forecast],
	);

	return (
		<div
			style={{
				padding: "16px 20px",
				backgroundColor: "#fff",
				borderRadius: "8px",
				border: "1px solid #e5e7eb",
			}}
		>
			<div
				style={{
					display: "flex",
					alignItems: "center",
					justifyContent: "space-between",
					marginBottom: "4px",
				}}
			>
				<h3 style={{ margin: 0, fontSize: "16px", fontWeight: 700 }}>
					Clogging Trajectory Forecast
				</h3>
				<div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
					<label style={{ fontSize: "12px", color: "#6b7280" }}>
						Critical level:
					</label>
					<select
						value={criticalMultiplier}
						onChange={(e) => onCriticalMultiplierChange(Number(e.target.value))}
						style={{
							fontSize: "12px",
							padding: "2px 6px",
							borderRadius: "4px",
							border: "1px solid #e5e7eb",
						}}
					>
						{MULTIPLIER_OPTIONS.map((v) => (
							<option key={v} value={v}>
								{v}×
							</option>
						))}
					</select>
				</div>
			</div>
			<p style={{ margin: "0 0 16px", fontSize: "13px", color: "#6b7280" }}>
				Onset detected at {formatHHMMSS(forecast.onset_time)} — fitting 3 growth
				models to project time to {forecast.critical_multiplier}× detection
				threshold ({forecast.critical_threshold.toFixed(6)})
			</p>

			{forecast.consensus_eta !== null ? (
				<div
					style={{
						marginBottom: "16px",
						padding: "12px 16px",
						backgroundColor: "#fef2f2",
						borderRadius: "8px",
						border: "1px solid #fca5a5",
						display: "flex",
						alignItems: "center",
						gap: "16px",
					}}
				>
					<div>
						<div style={{ fontSize: "11px", color: "#6b7280" }}>
							Consensus ETA (median of models)
						</div>
						<div
							style={{ fontSize: "22px", fontWeight: 700, color: "#dc2626" }}
						>
							{formatHHMMSS(forecast.consensus_eta)}
						</div>
					</div>
					<div style={{ fontSize: "12px", color: "#9ca3af", lineHeight: 1.5 }}>
						{forecast.post_onset_points.toLocaleString()} post-onset data points
						used for fitting
					</div>
				</div>
			) : (
				<div
					style={{
						marginBottom: "16px",
						padding: "12px 16px",
						backgroundColor: "#f0fdf4",
						borderRadius: "8px",
						border: "1px solid #86efac",
					}}
				>
					<div style={{ fontSize: "13px", color: "#166534" }}>
						<strong>No critical crossing projected.</strong> The composite score
						is not growing toward the {forecast.critical_multiplier}× threshold.
						The partial blockage appears stable — no progressive restriction
						growth detected.
					</div>
				</div>
			)}

			<div
				style={{
					display: "grid",
					gridTemplateColumns: "repeat(3, 1fr)",
					gap: "12px",
					marginBottom: "16px",
				}}
			>
				{Object.entries(forecast.fits).map(([name, fit]) => (
					<ModelFitCard
						key={name}
						name={name}
						fit={fit}
						isBest={name === forecast.best_fit}
						criticalMultiplier={forecast.critical_multiplier}
					/>
				))}
			</div>

			<div
				style={{
					fontSize: "13px",
					fontWeight: 600,
					color: "#374151",
					marginBottom: "8px",
				}}
			>
				Post-onset Trajectory + Growth Model Fits
			</div>
			<ResponsiveContainer width="100%" height={280}>
				<ComposedChart
					data={chartData}
					margin={{ top: 8, right: 24, left: 0, bottom: 0 }}
				>
					<CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
					<XAxis
						dataKey="time"
						tickFormatter={(v) => formatHHMMSS(Number(v))}
						stroke="#9ca3af"
						fontSize={11}
					/>
					<YAxis
						stroke="#9ca3af"
						fontSize={11}
						tickFormatter={(v) => Number(v).toFixed(4)}
					/>
					<Tooltip
						formatter={(v, name) => [Number(v).toFixed(6), name]}
						labelFormatter={(l) => `Time: ${formatHHMMSS(Number(l))}`}
					/>
					<Legend wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }} />
					<ReferenceLine
						y={forecast.critical_threshold}
						stroke="#dc2626"
						strokeDasharray="6 3"
						label={{
							value: `${forecast.critical_multiplier}× threshold`,
							position: "insideTopRight",
							fontSize: 11,
							fill: "#dc2626",
						}}
					/>
					<ReferenceLine
						y={forecast.detection_threshold}
						stroke="#f59e0b"
						strokeDasharray="4 3"
						label={{
							value: "Detection",
							position: "insideBottomRight",
							fontSize: 11,
							fill: "#d97706",
						}}
					/>
					<Line
						type="monotone"
						dataKey="actual"
						name="Actual composite"
						stroke="#8b5cf6"
						strokeWidth={1.5}
						dot={false}
						connectNulls
						isAnimationActive={false}
					/>
					{Object.keys(MODEL_COLORS).map((name) => (
						<Line
							key={name}
							type="monotone"
							dataKey={name}
							name={MODEL_LABELS[name]}
							stroke={MODEL_COLORS[name]}
							strokeWidth={1.5}
							strokeDasharray="5 4"
							dot={false}
							connectNulls
							isAnimationActive={false}
						/>
					))}
				</ComposedChart>
			</ResponsiveContainer>
		</div>
	);
}
