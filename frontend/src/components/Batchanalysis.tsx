import axios from "axios";
import { useCallback, useMemo, useState } from "react";
import { computeForecast, type ForecastData } from "../utils/forecastClient";
import type { LogEntry } from "../utils/resultLogStore";
import {
	appendEntry,
	clearLog,
	exportCsv,
	importFromCsv,
	loadLog,
	updateEntryActualTime,
} from "../utils/resultLogStore";
import type { SigmaValue } from "../utils/sigmaComparisonStore";
import {
	computeSigmaForecasts,
	loadStore,
} from "../utils/sigmaComparisonStore";
import {
	type AnalysisPoint,
	BatchMLSection,
	CrossingCard,
	SummaryCard,
} from "./BatchAnalysisCards";
import { CloggingForecast } from "./CloggingForecast";
import { ControlChart, type ControlChartDataPoint } from "./ControlChart";
import { LogTable } from "./LogTable";
import { SigmaComparisonChart } from "./SigmaComparisonChart";
import { SigmaErrorChart } from "./SigmaErrorChart";
import { Tooltip } from "./ui/Tooltip";

export interface CalibrationStats {
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

export interface AnalysisResult {
	timeseries: AnalysisPoint[];
	columns: string[];
	calibration: CalibrationStats;
	thresholds: Thresholds;
	forecast: ForecastData | null;
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

// ── Pure helpers (used both in useMemo and runAnalysis) ────────────────────

interface CrossingTimes {
	static: number | null;
	composite: number | null;
}

interface ThresholdSet {
	fft_threshold: number;
	static_threshold: number;
}

function computeCrossings(
	analysisPoints: AnalysisPoint[],
	thresh: ThresholdSet,
	smoothingWindowSec: number,
): CrossingTimes {
	let staticCrossing: number | null = null;
	for (const p of analysisPoints) {
		if (p.static_score > thresh.static_threshold) {
			staticCrossing = p.time;
			break;
		}
	}

	let compositeCrossing: number | null = null;
	let wStart = 0;
	for (let i = 0; i < analysisPoints.length; i++) {
		while (
			analysisPoints[wStart].time <
			analysisPoints[i].time - smoothingWindowSec
		)
			wStart++;
		const win = analysisPoints
			.slice(wStart, i + 1)
			.map((p) => p.composite_score);
		win.sort((a, b) => a - b);
		const mid = Math.floor(win.length / 2);
		const median =
			win.length % 2 === 1 ? win[mid] : (win[mid - 1] + win[mid]) / 2;
		if (median > thresh.fft_threshold) {
			compositeCrossing = analysisPoints[i].time;
			break;
		}
	}

	return { static: staticCrossing, composite: compositeCrossing };
}

function secToHHMMSS(seconds: number | null): string | null {
	if (seconds === null) return null;
	const h = Math.floor(seconds / 3600);
	const m = Math.floor((seconds % 3600) / 60);
	const s = Math.floor(seconds % 60);
	return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
}

interface LogEntryInput {
	result: AnalysisResult;
	sigma: number;
	calibrationSeconds: number;
	smoothingWindowSec: number;
	crossings: CrossingTimes;
	sigmaForecasts: ReturnType<typeof computeSigmaForecasts>;
	forecast: ForecastData | null;
	peakMlProb: number | null;
	actualCloggingTime: number | null;
}

function buildLogEntry(input: LogEntryInput): LogEntry {
	const {
		result,
		sigma,
		calibrationSeconds,
		smoothingWindowSec,
		crossings,
		sigmaForecasts,
		forecast,
		peakMlProb,
		actualCloggingTime,
	} = input;
	const basename =
		result.metadata.file.split(/[\\/]/).pop() ?? result.metadata.file;
	return {
		filename: basename,
		timestamp: new Date().toISOString(),
		sigma,
		calibrationSeconds,
		smoothingWindowSec,
		durationMin: parseFloat((result.metadata.duration_seconds / 60).toFixed(2)),
		totalPoints: result.metadata.total_points,
		samplingHz: result.metadata.sampling_hz,
		compositeThreshold: result.thresholds.fft_threshold,
		staticThreshold: result.thresholds.static_threshold,
		compositeCrossing: secToHHMMSS(crossings.composite),
		staticCrossing: secToHHMMSS(crossings.static),
		forecastOnset: forecast ? secToHHMMSS(forecast.onset_time) : null,
		forecastEtaSigma3: secToHHMMSS(
			sigmaForecasts[3 as SigmaValue].consensusEta,
		),
		forecastEtaSigma4: secToHHMMSS(
			sigmaForecasts[4 as SigmaValue].consensusEta,
		),
		forecastEtaSigma5: secToHHMMSS(
			sigmaForecasts[5 as SigmaValue].consensusEta,
		),
		actualCloggingTime: secToHHMMSS(actualCloggingTime),
		bestFitModel: forecast?.best_fit ?? null,
		bestFitR2: forecast ? (forecast.fits[forecast.best_fit]?.r2 ?? null) : null,
		peakMlProbability: peakMlProb,
	};
}

export function BatchAnalysis({ selectedFile }: BatchAnalysisProps) {
	const [result, setResult] = useState<AnalysisResult | null>(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [sigma, setSigma] = useState(3.0);
	const [thresholds, setThresholds] = useState<Thresholds | null>(null);
	// 0 ⇒ match real-time exactly (omit calibration_seconds → backend uses dynamic sample count)
	const [calibrationSeconds, setCalibrationSeconds] = useState(0);
	const [smoothingWindowSec, setSmoothingWindowSec] = useState(60);
	const [criticalMultiplier, setCriticalMultiplier] = useState(2.0);
	const [logEntries, setLogEntries] = useState<LogEntry[]>(() => loadLog());

	const logAnalysisResult = useCallback(
		(
			data: AnalysisResult,
			usedSigma: number,
			usedCalibration: number,
			usedWindow: number,
			usedMultiplier: number,
		) => {
			const analysisPoints = data.timeseries.filter(
				(p) => p.phase === "analysis",
			);
			const crossings = computeCrossings(
				analysisPoints,
				data.thresholds,
				usedWindow,
			);
			const sigmaForecasts = computeSigmaForecasts(data, usedWindow);
			const forecast = computeForecast(
				data.timeseries,
				data.thresholds.fft_threshold,
				{
					criticalMultiplier: usedMultiplier,
					onsetTime: crossings.composite ?? undefined,
					smoothingWindowSec: usedWindow,
				},
			);
			const mlPoints = analysisPoints.filter((p) => p.ensemble_probability > 0);
			const peakMl = mlPoints.length
				? Math.max(...mlPoints.map((p) => p.ensemble_probability))
				: null;
			const actualCloggingTime =
				loadStore()[data.metadata.file]?.actualCloggingTime ?? null;

			appendEntry(
				buildLogEntry({
					result: data,
					sigma: usedSigma,
					calibrationSeconds: usedCalibration,
					smoothingWindowSec: usedWindow,
					crossings,
					sigmaForecasts,
					forecast,
					peakMlProb: peakMl,
					actualCloggingTime,
				}),
			);
		},
		[],
	);

	// Run full analysis
	const runAnalysis = useCallback(async () => {
		if (!selectedFile) return;
		setLoading(true);
		setError(null);

		try {
			const calibParam =
				calibrationSeconds > 0
					? `&calibration_seconds=${calibrationSeconds}`
					: "";
			const response = await axios.post<AnalysisResult>(
				`/api/analyze?file=${encodeURIComponent(selectedFile)}&sigma=${sigma}${calibParam}`,
			);
			setResult(response.data);
			setThresholds(response.data.thresholds);
			logAnalysisResult(
				response.data,
				sigma,
				calibrationSeconds,
				smoothingWindowSec,
				criticalMultiplier,
			);
			setLogEntries(loadLog());
		} catch (err: unknown) {
			const e = err as {
				response?: { data?: { detail?: string } };
				message?: string;
			};
			setError(e.response?.data?.detail ?? e.message ?? "Analysis failed");
		} finally {
			setLoading(false);
		}
	}, [
		selectedFile,
		sigma,
		calibrationSeconds,
		smoothingWindowSec,
		criticalMultiplier,
		logAnalysisResult,
	]);

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
		if (!result)
			return { flow: [], static_: [], composite: [], compositeSmoothed: [] };

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
				compositeSmoothed.push({
					time: p.time,
					value: sm ?? p.composite_score,
				});
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
	// Static: first raw sample above threshold.
	// Composite: first rolling-median crossing (robust to transient spikes).
	const thresholdCrossings = useMemo(() => {
		if (!result)
			return {
				static: null as number | null,
				composite: null as number | null,
			};
		const thresh = thresholds ?? {
			sigma: 3.0,
			fft_threshold: 0.05,
			static_threshold: 0.018,
		};
		const analysisPoints = result.timeseries.filter(
			(p) => p.phase === "analysis",
		);
		return computeCrossings(analysisPoints, thresh, smoothingWindowSec);
	}, [result, thresholds, smoothingWindowSec]);

	const effectiveThresholds = thresholds || {
		sigma: 3.0,
		fft_threshold: 0.05,
		static_threshold: 0.018,
	};

	// Forecast recomputes whenever sigma/multiplier/smoothing/onset changes — no server round-trip needed
	const activeForecast = useMemo((): ForecastData | null => {
		if (!result) return null;
		return computeForecast(
			result.timeseries,
			effectiveThresholds.fft_threshold,
			{
				criticalMultiplier,
				onsetTime: thresholdCrossings.composite ?? undefined,
				smoothingWindowSec,
			},
		);
	}, [
		result,
		effectiveThresholds.fft_threshold,
		criticalMultiplier,
		thresholdCrossings.composite,
		smoothingWindowSec,
	]);

	// ── ML analytics (analysis phase only) ────────────────────────────────────
	const trafficDist = useMemo(() => {
		if (!result) return null;
		const pts = result.timeseries.filter((p) => p.phase === "analysis");
		if (!pts.length) return null;
		const total = pts.length;
		return {
			green: pts.filter((p) => p.traffic_light === "green").length / total,
			yellow: pts.filter((p) => p.traffic_light === "yellow").length / total,
			red: pts.filter((p) => p.traffic_light === "red").length / total,
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

	// Collect all distinct model names present in the result
	const modelNames = useMemo(() => {
		if (!result) return [] as string[];
		const names = new Set<string>();
		for (const p of result.timeseries) {
			if (p.models) Object.keys(p.models).forEach((k) => names.add(k));
		}
		return Array.from(names);
	}, [result]);

	const mlChartData = useMemo(() => {
		if (!result) return [];
		const pts = result.timeseries.filter((p) => p.phase === "analysis");
		const maxPts = 2000;
		const step = Math.max(1, Math.floor(pts.length / maxPts));
		return pts
			.filter((_, i) => i % step === 0)
			.map((p) => {
				const point: Record<string, number> = {
					time: p.time,
					ensemble: p.ensemble_probability,
				};
				if (p.models) {
					Object.entries(p.models).forEach(([name, pred]) => {
						point[name] = pred.probability;
					});
				}
				return point;
			});
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
							<label
								style={{ fontSize: "13px", color: "#6b7280" }}
								title="0 = match real-time exactly (uses dynamic per-fs calibration). Set >0 to override."
							>
								Calibration seconds:
							</label>
							<input
								type="number"
								value={calibrationSeconds}
								onChange={(e) =>
									setCalibrationSeconds(Math.max(0, Number(e.target.value)))
								}
								min={0}
								max={300}
								step={5}
								title="0 = match real-time"
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
						<div
							style={{
								display: "flex",
								alignItems: "center",
								gap: "6px",
								borderLeft: "1px solid #e5e7eb",
								paddingLeft: "12px",
							}}
						>
							<span style={{ fontSize: "12px", color: "#6b7280" }}>
								{logEntries.length} logged
							</span>
							<button
								onClick={() => {
									exportCsv();
								}}
								disabled={logEntries.length === 0}
								title="Download all logged results as CSV"
								style={{
									padding: "6px 12px",
									borderRadius: "6px",
									border: "1px solid #e5e7eb",
									background: logEntries.length === 0 ? "#f9fafb" : "#fff",
									color: logEntries.length === 0 ? "#9ca3af" : "#374151",
									cursor: logEntries.length === 0 ? "not-allowed" : "pointer",
									fontSize: "12px",
									fontWeight: 500,
								}}
							>
								Export CSV
							</button>
							<button
								onClick={() => {
									if (
										window.confirm(
											`Clear all ${logEntries.length} log entries? This cannot be undone.`,
										)
									) {
										clearLog();
										setLogEntries([]);
									}
								}}
								disabled={logEntries.length === 0}
								title="Clear all logged results"
								style={{
									padding: "6px 10px",
									borderRadius: "6px",
									border: `1px solid ${logEntries.length === 0 ? "#e5e7eb" : "#fca5a5"}`,
									background: "#fff",
									color: logEntries.length === 0 ? "#9ca3af" : "#dc2626",
									cursor: logEntries.length === 0 ? "not-allowed" : "pointer",
									fontSize: "12px",
								}}
							>
								Clear log
							</button>
						</div>
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
								<Tooltip text="Detection sensitivity. Lower σ = more sensitive (more alerts). Higher σ = stricter (fewer alerts). 3σ is a good starting point.">
									Threshold σ:
								</Tooltip>
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
							<span
								style={{ fontSize: "13px", fontWeight: 600, color: "#374151" }}
							>
								Sigma Threshold Crossings ({sigma}σ)
							</span>
							<div
								style={{ display: "flex", alignItems: "center", gap: "6px" }}
							>
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
						<div
							style={{
								display: "grid",
								gridTemplateColumns: "1fr 1fr",
								gap: "12px",
							}}
						>
							<CrossingCard
								label="Static — first raw sample above threshold"
								time={thresholdCrossings.static}
								threshold={effectiveThresholds.static_threshold}
								color="#2E7D32"
							/>
							<CrossingCard
								label={`Composite — rolling median (${smoothingWindowSec}s) crosses threshold`}
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

					{/* ── Clogging Forecast — updates live when sigma changes ── */}
					{activeForecast && (
						<CloggingForecast
							forecast={activeForecast}
							timeseries={result.timeseries}
							criticalMultiplier={criticalMultiplier}
							onCriticalMultiplierChange={setCriticalMultiplier}
						/>
					)}

					{/* ── Sigma Comparison — predicted vs actual, accumulates across files ── */}
					{selectedFile && (
						<SigmaComparisonChart
							result={result}
							selectedFile={selectedFile}
							smoothingWindowSec={smoothingWindowSec}
						/>
					)}

					{/* ── ML Results Section ── */}
					{(trafficDist || mlStats) && (
						<BatchMLSection
							trafficDist={trafficDist}
							mlStats={mlStats}
							mlChartData={mlChartData}
							modelNames={modelNames}
							timeseries={result.timeseries}
						/>
					)}
				</div>
			)}

			{/* ── Analysis Log ── */}
			{logEntries.length > 0 && (
				<details
					style={{
						marginTop: "20px",
						padding: "16px 20px",
						backgroundColor: "#fff",
						borderRadius: "8px",
						border: "1px solid #e5e7eb",
					}}
				>
					<summary
						style={{
							cursor: "pointer",
							fontWeight: 600,
							fontSize: "14px",
							userSelect: "none",
						}}
					>
						Analysis Log
						<span
							style={{
								fontWeight: 400,
								fontSize: "12px",
								color: "#6b7280",
								marginLeft: "8px",
							}}
						>
							{logEntries.length}{" "}
							{logEntries.length === 1 ? "entry" : "entries"} — click to expand
						</span>
					</summary>
					<div style={{ marginTop: "14px" }}>
						<LogTable
							entries={logEntries}
							onUpdateActualTime={(entry, value) => {
								updateEntryActualTime(entry.timestamp, entry.filename, value);
								setLogEntries(loadLog());
							}}
							onImport={(csvText) => {
								const count = importFromCsv(csvText);
								setLogEntries(loadLog());
								if (count === 0)
									alert(
										"No new entries found (already imported or invalid format).",
									);
								else alert(`Imported ${count} new entries.`);
							}}
						/>
						<SigmaErrorChart entries={logEntries} />
					</div>
				</details>
			)}
		</div>
	);
}
