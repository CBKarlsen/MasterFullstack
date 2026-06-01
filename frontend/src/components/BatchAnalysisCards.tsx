/**
 * BatchAnalysisCards — presentational building blocks for the BATCH-mode results view.
 *
 * This module groups the small, reusable result cards and the ML results panel that
 * BatchAnalysis composes after a file has been analysed. It exports the shared
 * `AnalysisPoint` type (one scored sample, carrying the four detector scores, the
 * traffic-light state, the ML ensemble probability and optional per-model
 * predictions, plus its calibration/analysis `phase`), the simple display cards
 * (`SummaryCard`, `CrossingCard`, `MLStatCard`), and `BatchMLSection`, which renders
 * the signal-state distribution bar, aggregate ML stat cards, a per-model breakdown
 * (highlighting models that are still untrained/"dormant"), and a probability-over-time
 * chart. These components are purely presentational — all scoring, phase splitting and
 * threshold logic live in BatchAnalysis and the backend; here data only arrives as props.
 */
import { useMemo } from "react";
import {
	Area,
	CartesianGrid,
	ComposedChart,
	Legend,
	Line,
	ReferenceArea,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export interface ModelPrediction {
	probability: number;
	confidence: number;
	classification?: string;
	weight?: number;
}

export interface AnalysisPoint {
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
	models?: Record<string, ModelPrediction>;
	phase: string;
	raw?: Record<string, number>;
}

export interface TrafficDist {
	green: number;
	yellow: number;
	red: number;
	total: number;
}

export interface MLStats {
	peak: number;
	firstCriticalSec: number | null;
	avgRedProb: number | null;
}

// Dynamic: includes ensemble + one key per active model
export type MLChartPoint = Record<string, number>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Format a seconds value as a zero-padded HH:MM:SS string.
function formatHHMMSS(seconds: number): string {
	const h = Math.floor(seconds / 3600);
	const m = Math.floor((seconds % 3600) / 60);
	const s = Math.floor(seconds % 60);
	return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
}

const MODEL_COLORS: Record<string, string> = {
	"FFT Physics": "#f59e0b",
	"Random Forest": "#16a34a",
	"Isolation Forest": "#0284c7",
};
const FALLBACK_COLORS = ["#8b5cf6", "#ec4899", "#06b6d4", "#f97316"];

// Resolve a stable colour for a model: fixed colour by name, else a fallback by index.
function modelColor(name: string, idx: number): string {
	return MODEL_COLORS[name] ?? FALLBACK_COLORS[idx % FALLBACK_COLORS.length];
}

// ---------------------------------------------------------------------------
// SummaryCard
// ---------------------------------------------------------------------------

interface SummaryCardProps {
	label: string;
	value: string;
	highlight?: boolean;
}

// Single label/value metric tile for the top summary bar; `highlight` styles it red.
export function SummaryCard({
	label,
	value,
	highlight = false,
}: SummaryCardProps) {
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

// ---------------------------------------------------------------------------
// CrossingCard
// ---------------------------------------------------------------------------

interface CrossingCardProps {
	label: string;
	time: number | null;
	threshold: number;
	color: string;
}

// Shows a single threshold-crossing time (or "No crossing") with its threshold value.
export function CrossingCard({
	label,
	time,
	threshold,
	color,
}: CrossingCardProps) {
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

// ---------------------------------------------------------------------------
// MLStatCard
// ---------------------------------------------------------------------------

interface MLStatCardProps {
	label: string;
	value: string;
	highlight?: boolean;
}

// Compact label/value tile used for aggregate ML statistics.
export function MLStatCard({
	label,
	value,
	highlight = false,
}: MLStatCardProps) {
	return (
		<div
			style={{
				padding: "12px 14px",
				backgroundColor: highlight ? "#fef2f2" : "#f9fafb",
				borderRadius: "8px",
				border: `1px solid ${highlight ? "#fca5a5" : "#e5e7eb"}`,
			}}
		>
			<div style={{ fontSize: "11px", color: "#6b7280", marginBottom: "4px" }}>
				{label}
			</div>
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

// ---------------------------------------------------------------------------
// BatchMLSection
// ---------------------------------------------------------------------------

interface BatchMLSectionProps {
	trafficDist: TrafficDist | null;
	mlStats: MLStats | null;
	mlChartData: MLChartPoint[];
	modelNames: string[];
	timeseries: AnalysisPoint[];
}

const CRITICAL_PROBABILITY_THRESHOLD = 0.7;
const DORMANT_MIN_POINTS = 10;

// ML results panel: traffic-light distribution bar, aggregate ML stat cards, a
// per-model breakdown, and the probability-over-time chart. A model is treated as
// "dormant" (untrained) when it never produced a non-zero probability across a
// sufficiently long run, and is then shown as awaiting training and hidden from the chart.
export function BatchMLSection({
	trafficDist,
	mlStats,
	mlChartData,
	modelNames,
	timeseries,
}: BatchMLSectionProps) {
	const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

	// Per model: peak probability, first time it crossed the critical level, and a
	// dormant flag (no signal at all over a long enough run ⇒ model not yet trained).
	const perModelStats = useMemo(() => {
		const pts = timeseries.filter((p) => p.phase === "analysis" && p.models);
		return modelNames.map((name, idx) => {
			let peak = 0;
			let firstCriticalSec: number | null = null;
			for (const p of pts) {
				const prob = p.models?.[name]?.probability ?? 0;
				if (prob > peak) peak = prob;
				if (firstCriticalSec === null && prob >= CRITICAL_PROBABILITY_THRESHOLD)
					firstCriticalSec = p.time;
			}
			const dormant = peak === 0 && pts.length > DORMANT_MIN_POINTS;
			return {
				name,
				peak,
				firstCriticalSec,
				color: modelColor(name, idx),
				dormant,
			};
		});
	}, [modelNames, timeseries]);

	return (
		<div style={{ marginTop: "20px" }}>
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

				{trafficDist && (
					<div style={{ marginBottom: "16px" }}>
						<div
							style={{
								fontSize: "12px",
								color: "#6b7280",
								marginBottom: "6px",
								fontWeight: 500,
							}}
						>
							Signal State Distribution ({trafficDist.total.toLocaleString()}{" "}
							analysis samples)
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
									style={{
										flex: trafficDist.green,
										backgroundColor: "#22c55e",
									}}
								/>
							)}
							{trafficDist.yellow > 0.005 && (
								<div
									title={`Warning: ${pct(trafficDist.yellow)}`}
									style={{
										flex: trafficDist.yellow,
										backgroundColor: "#f59e0b",
									}}
								/>
							)}
							{trafficDist.red > 0.005 && (
								<div
									title={`Clogged: ${pct(trafficDist.red)}`}
									style={{ flex: trafficDist.red, backgroundColor: "#ef4444" }}
								/>
							)}
						</div>
						<div
							style={{
								display: "flex",
								gap: "20px",
								marginTop: "6px",
								fontSize: "12px",
							}}
						>
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
							highlight={mlStats.peak >= CRITICAL_PROBABILITY_THRESHOLD}
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
							value={
								mlStats.avgRedProb !== null ? pct(mlStats.avgRedProb) : "—"
							}
							highlight={
								(mlStats.avgRedProb ?? 0) >= CRITICAL_PROBABILITY_THRESHOLD
							}
						/>
					</div>
				)}
			</div>

			{perModelStats.length > 0 && (
				<div
					style={{
						padding: "16px 20px",
						backgroundColor: "#fff",
						borderRadius: "8px",
						border: "1px solid #e5e7eb",
						marginBottom: "12px",
					}}
				>
					<div
						style={{
							fontSize: "14px",
							fontWeight: 600,
							color: "#374151",
							marginBottom: "12px",
						}}
					>
						Per-Model Breakdown
					</div>
					<div
						style={{
							display: "grid",
							gridTemplateColumns: `repeat(${Math.min(perModelStats.length, 3)}, 1fr)`,
							gap: "12px",
						}}
					>
						{perModelStats.map(
							({ name, peak, firstCriticalSec, color, dormant }) => (
								<div
									key={name}
									style={{
										padding: "12px 14px",
										borderRadius: "8px",
										border: `2px solid ${dormant ? "#fcd34d" : color + "20"}`,
										backgroundColor: dormant ? "#fffbeb" : `${color}08`,
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
												backgroundColor: dormant ? "#fbbf24" : color,
												flexShrink: 0,
											}}
										/>
										<span
											style={{
												fontSize: "13px",
												fontWeight: 700,
												color: "#374151",
											}}
										>
											{name}
										</span>
										{dormant && (
											<span
												style={{
													fontSize: "10px",
													fontWeight: 700,
													padding: "2px 6px",
													backgroundColor: "#fef3c7",
													color: "#92400e",
													borderRadius: "10px",
													border: "1px solid #fcd34d",
													marginLeft: "auto",
												}}
											>
												Awaiting training
											</span>
										)}
									</div>
									{dormant ? (
										<div
											style={{
												fontSize: "11px",
												color: "#92400e",
												lineHeight: 1.5,
											}}
										>
											Not yet trained. Go to the <strong>Models</strong> tab to
											train this model.
										</div>
									) : (
										<div
											style={{
												display: "flex",
												flexDirection: "column",
												gap: "4px",
												fontSize: "12px",
											}}
										>
											<div
												style={{
													display: "flex",
													justifyContent: "space-between",
												}}
											>
												<span style={{ color: "#6b7280" }}>
													Peak probability
												</span>
												<span
													style={{
														fontWeight: 700,
														color:
															peak >= 0.7
																? "#dc2626"
																: peak >= 0.3
																	? "#d97706"
																	: "#16a34a",
													}}
												>
													{pct(peak)}
												</span>
											</div>
											<div
												style={{
													display: "flex",
													justifyContent: "space-between",
												}}
											>
												<span style={{ color: "#6b7280" }}>
													First critical ≥70%
												</span>
												<span
													style={{
														fontWeight: 600,
														color:
															firstCriticalSec !== null ? "#dc2626" : "#9ca3af",
													}}
												>
													{firstCriticalSec !== null
														? formatHHMMSS(firstCriticalSec)
														: "None"}
												</span>
											</div>
											<div
												style={{
													marginTop: "4px",
													height: "6px",
													backgroundColor: "#e5e7eb",
													borderRadius: "3px",
													overflow: "hidden",
												}}
											>
												<div
													style={{
														height: "100%",
														width: `${peak * 100}%`,
														backgroundColor: color,
														borderRadius: "3px",
													}}
												/>
											</div>
										</div>
									)}
								</div>
							),
						)}
					</div>
				</div>
			)}

			{mlChartData.length > 0 && (
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
							fontSize: "14px",
							fontWeight: 600,
							color: "#374151",
							marginBottom: "12px",
						}}
					>
						ML Probability Over Time
					</div>
					<ResponsiveContainer width="100%" height={260}>
						<ComposedChart
							data={mlChartData}
							margin={{ top: 8, right: 20, left: 0, bottom: 0 }}
						>
							<CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
							<ReferenceArea
								y1={0}
								y2={0.2}
								fill="#22c55e"
								fillOpacity={0.07}
								ifOverflow="hidden"
							/>
							<ReferenceArea
								y1={0.2}
								y2={0.7}
								fill="#f59e0b"
								fillOpacity={0.07}
								ifOverflow="hidden"
							/>
							<ReferenceArea
								y1={0.7}
								y2={1}
								fill="#ef4444"
								fillOpacity={0.07}
								ifOverflow="hidden"
							/>
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
								formatter={(v, name) => [
									`${(Number(v) * 100).toFixed(1)}%`,
									name,
								]}
								labelFormatter={(l) => `Time: ${Number(l).toFixed(1)}s`}
							/>
							<Legend wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }} />
							{perModelStats
								.filter(({ dormant }) => !dormant)
								.map(({ name, color }) => (
									<Line
										key={name}
										type="monotone"
										dataKey={name}
										name={name}
										stroke={color}
										strokeWidth={1.5}
										strokeDasharray="4 3"
										dot={false}
										isAnimationActive={false}
									/>
								))}
							<Area
								type="monotone"
								dataKey="ensemble"
								name="Ensemble"
								stroke="#8b5cf6"
								strokeWidth={2.5}
								fill="#8b5cf6"
								fillOpacity={0.12}
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
