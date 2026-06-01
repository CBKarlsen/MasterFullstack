/**
 * Dashboard — the top-level screen of the pipe-clogging detection platform.
 *
 * Renders the application shell (header, mode tabs, status pill, Start/Stop
 * control) and lays out the live monitoring view: traffic light, ensemble
 * clogging probability, per-method metric boxes (spectral slope, static and
 * composite scores), the probability/raw-data/control charts, the live ETA
 * forecast panel, and the right-hand sidebar (file/column selectors, model
 * upload, ML live panel).
 *
 * The component owns only UI-local state (selected file, playback speed, and
 * the active mode tab: "realtime" | "batch" | "models"). All streamed sensor
 * and detection state lives in the Zustand `simulationStore`. It opens the
 * backend WebSocket via `useWebSocket` and wires the hook's batched-message
 * callback straight to the store's `updateDataBatch` action, so incoming
 * frames flow WebSocket -> store -> charts without passing through this
 * component's render.
 *
 * Note: the WebSocket connects directly to the backend in dev (bypassing
 * Vite's proxy, which drops WS connections); in production a relative path is
 * used so it rides the same origin.
 */
import axios from "axios";
import { useCallback, useEffect, useState } from "react";
import { useWebSocket } from "../hooks/useWebSocket";
import { useSimulationStore } from "../store/simulationStore";
import {
	COLOR,
	FONT_SIZE,
	FONT_WEIGHT,
	RADIUS,
	SHADOW,
	SPACING,
} from "../styles/tokens";
import type { TrafficLight as TLType } from "../types";
import { BatchAnalysis } from "./BatchAnalysis";
import { ColumnSelector } from "./ColumnSelector";
import { ControlChartGrid } from "./ControlChartGrid";
import { DataFileSelector } from "./DataFileSelector";
import { LiveForecastPanel } from "./LiveForecastPanel";
import { MLLivePanel } from "./MLLivePanel";
import { ModelsTab } from "./ModelsTab";
import { ModelUpload } from "./ModelUpload";
import { ProbabilityChart } from "./ProbabilityChart";
import { RawDataChart } from "./RawDataChart";
import { TrafficLight } from "./TrafficLight";
import { Button } from "./ui/Button";
import { Tooltip } from "./ui/Tooltip";

export function Dashboard() {
	const {
		currentData,
		status,
		chartData,
		rawChartData,
		models,
		isCalibrating,
		calibrationProgress,
		availableColumns,
		selectedColumns,
		updateDataBatch,
		setModels,
		toggleModel,
		setSelectedColumns,
		clearData,
	} = useSimulationStore();

	const [mode, setMode] = useState<"realtime" | "batch" | "models">("realtime");
	// Simulation configuration
	const [selectedFile, setSelectedFile] = useState<string | null>(null);
	const [speed, setSpeed] = useState<number>(1);

	// Connect directly to the backend, bypassing Vite's proxy which
	// drops WebSocket connections via ECONNRESET.
	const wsUrl = import.meta.env.DEV
		? "ws://localhost:8000/ws/simulate"
		: "/ws/simulate";

	// Passing `onBatch` (rather than a per-message handler) lets the hook hand
	// off a whole animation-frame's worth of samples in a single store write.
	const { isConnected, startSimulation, stopSimulation, error } = useWebSocket(
		wsUrl,
		{ onBatch: updateDataBatch },
	);

	const fetchModels = useCallback(async () => {
		try {
			const response = await axios.get("/api/models");
			setModels(response.data);
		} catch (error) {
			console.error("Failed to fetch models:", error);
		}
	}, [setModels]);

	// Fetch models on mount
	useEffect(() => {
		fetchModels();
	}, [fetchModels]);

	const handleStart = () => {
		if (!selectedFile) {
			alert("Please select a data file first");
			return;
		}
		// Reset store buffers before a new run so stale chart history is dropped.
		clearData();
		startSimulation(selectedFile, speed);
	};

	const handleStop = () => {
		stopSimulation();
	};

	return (
		<div style={{ padding: SPACING.XXL, maxWidth: "1400px", margin: "0 auto" }}>
			{/* Header */}
			<div
				style={{
					display: "flex",
					justifyContent: "space-between",
					alignItems: "center",
					marginBottom: SPACING.XXL,
				}}
			>
				<div>
					<h1 style={{ margin: 0, fontSize: FONT_SIZE.XXL }}>
						Pipe Clogging Detection Platform
					</h1>
					<div
						style={{
							display: "flex",
							gap: SPACING.XS,
							background: COLOR.GRAY_100,
							borderRadius: RADIUS.LG,
							padding: "3px",
						}}
					>
						{(["realtime", "batch", "models"] as const).map((tab) => (
							<button
								key={tab}
								onClick={() => setMode(tab)}
								style={{
									padding: `${SPACING.XS} ${SPACING.LG}`,
									borderRadius: RADIUS.MD,
									border: "none",
									background: mode === tab ? COLOR.WHITE : "transparent",
									fontWeight:
										mode === tab ? FONT_WEIGHT.SEMIBOLD : FONT_WEIGHT.NORMAL,
									cursor: "pointer",
									fontSize: FONT_SIZE.MD,
									boxShadow: mode === tab ? SHADOW.SM : "none",
									fontFamily: "inherit",
								}}
							>
								{tab === "realtime"
									? "Real-time"
									: tab === "batch"
										? "Batch Analysis"
										: "Models"}
							</button>
						))}
					</div>
					<p style={{ margin: `${SPACING.XS} 0 0`, color: COLOR.GRAY_500 }}>
						Real-time FFT & ML-based anomaly detection
					</p>
				</div>
				<div style={{ display: "flex", gap: SPACING.MD, alignItems: "center" }}>
					{/* Speed selector */}
					<div
						style={{ display: "flex", alignItems: "center", gap: SPACING.SM }}
					>
						<label style={{ fontSize: FONT_SIZE.BASE, color: COLOR.GRAY_500 }}>
							Speed:
						</label>
						{/* Playback speed multiplier for the simulated stream; locked
						    while a run is in progress (cannot change mid-stream). */}
						<select
							value={speed}
							onChange={(e) => setSpeed(Number(e.target.value))}
							disabled={isConnected}
							style={{
								padding: `${SPACING.XS} ${SPACING.SM}`,
								borderRadius: RADIUS.SM,
								border: `1px solid ${COLOR.GRAY_200}`,
								fontSize: FONT_SIZE.BASE,
							}}
						>
							<option value={0.5}>0.5x</option>
							<option value={1}>1x (Real-time)</option>
							<option value={2}>2x</option>
							<option value={5}>5x</option>
							<option value={10}>10x</option>
							<option value={20}>20x</option>
							<option value={50}>50x</option>
						</select>
					</div>

					<span
						style={{
							padding: `${SPACING.XS} ${SPACING.MD}`,
							borderRadius: RADIUS.PILL,
							fontSize: FONT_SIZE.BASE,
							backgroundColor: isConnected ? COLOR.SUCCESS_BG : "#fee2e2",
							color: isConnected ? COLOR.SUCCESS_TEXT : COLOR.DANGER_TEXT,
						}}
					>
						{isConnected ? "Running" : "Stopped"}
					</span>
					<Button
						onClick={isConnected ? handleStop : handleStart}
						variant={isConnected ? "danger" : "primary"}
						disabled={!selectedFile && !isConnected}
					>
						{isConnected ? "Stop" : "Start Simulation"}
					</Button>
				</div>
			</div>

			{error && (
				<div
					style={{
						padding: SPACING.MD,
						marginBottom: SPACING.LG,
						backgroundColor: "#fee2e2",
						color: COLOR.DANGER_TEXT,
						borderRadius: RADIUS.MD,
					}}
				>
					{error}
				</div>
			)}

			{/* Models tab — full-width, no sidebar */}
			{mode === "models" && <ModelsTab />}

			{/* Main Grid — realtime + batch only. Kept mounted (display:none)
			    rather than unmounted in models mode so its state survives a tab
			    switch; the left column then forks on realtime vs. batch. */}
			<div
				style={{
					display: mode === "models" ? "none" : "grid",
					gridTemplateColumns: "1fr 350px",
					gap: "24px",
				}}
			>
				{/* Left Column - Charts and Status */}
				<div style={{ minWidth: 0 }}>
					{/* Status Card */}
					{mode === "realtime" ? (
						<>
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
									<TrafficLight
										color={(currentData?.traffic_light || "gray") as TLType}
										message={currentData?.light_msg || status}
									/>
									<div style={{ textAlign: "right" }}>
										<div style={{ fontSize: "32px", fontWeight: "bold" }}>
											{/* Ensemble clogging probability (0–1) shown as a percent;
											    em-dash until the first post-calibration sample arrives. */}
											{currentData?.ensemble_probability !== undefined
												? `${(currentData.ensemble_probability * 100).toFixed(1)}%`
												: "—"}
										</div>
										<div style={{ fontSize: "14px", color: "#6b7280" }}>
											<Tooltip text="Combined probability of a clogging event across all active ML models.">
												Clogging Probability
											</Tooltip>
										</div>
									</div>
								</div>

								{/* Current Values */}
								{currentData && currentData.time !== undefined && (
									<div
										style={{
											display: "grid",
											gridTemplateColumns: "repeat(4, 1fr)",
											gap: "16px",
											marginTop: "20px",
											paddingTop: "16px",
											borderTop: "1px solid #e5e7eb",
										}}
									>
										<MetricBox
											label="Time"
											value={`${currentData.time?.toFixed(1)}s`}
										/>
										<MetricBox
											label="Spectral Slope"
											tooltip="Log-log slope of the power spectrum. Healthy flow: ~−2.5. A flatter slope (closer to 0) indicates turbulence from a blockage."
											value={currentData.spectral_slope?.toFixed(2) || "—"}
										/>
										<MetricBox
											label="Static Score"
											tooltip="Sustained deviation of mean pressure from the healthy baseline. Detects full blockages but is slow to react to partial ones."
											value={currentData.static_score?.toFixed(4) || "—"}
										/>
										<MetricBox
											label="Composite Score"
											tooltip="L1 distance between the current FFT spectrum and the healthy baseline. Best method for detecting partial blockages early."
											value={currentData.composite_score?.toFixed(4) || "—"}
										/>
									</div>
								)}
							</div>

							{/* Probability Chart */}
							<div
								style={{
									padding: "20px",
									backgroundColor: "#fff",
									borderRadius: "8px",
									border: "1px solid #e5e7eb",
								}}
							>
								<h2 style={{ margin: "0 0 16px", fontSize: "18px" }}>
									Model Probabilities Over Time
								</h2>
								<ProbabilityChart
									data={chartData}
									models={models}
									isCalibrating={isCalibrating}
									calibrationProgress={calibrationProgress}
								/>
							</div>

							{/* Raw Data Chart */}
							<div
								style={{
									padding: "20px",
									backgroundColor: "#fff",
									borderRadius: "8px",
									border: "1px solid #e5e7eb",
									marginTop: "20px",
								}}
							>
								<h2 style={{ margin: "0 0 16px", fontSize: "18px" }}>
									Physics Data
								</h2>
								<RawDataChart
									data={rawChartData}
									selectedColumns={selectedColumns}
									height={250}
								/>
							</div>

							{/* Control Charts Grid */}
							<div style={{ marginTop: "20px" }}>
								<ControlChartGrid
									rawData={rawChartData}
									selectedColumns={selectedColumns}
									chartData={chartData}
									maxDataPoints={2000}
									chartHeight={180}
								/>
							</div>

							{/* Live Forecast — appears automatically once threshold is crossed */}
							<LiveForecastPanel
								chartData={chartData}
								fftThreshold={currentData?.limit_threshold}
							/>
						</>
					) : (
						<BatchAnalysis selectedFile={selectedFile} />
					)}
				</div>

				{/* Right Column - Data & Models */}
				<div>
					{/* Data File Selector */}
					<div style={{ marginBottom: "20px" }}>
						<DataFileSelector
							selectedFile={selectedFile}
							onSelectFile={setSelectedFile}
							disabled={isConnected}
						/>
					</div>

					{/* Column Selector */}
					<div style={{ marginBottom: "20px" }}>
						<ColumnSelector
							availableColumns={availableColumns}
							selectedColumns={selectedColumns}
							onSelectionChange={setSelectedColumns}
							disabled={false}
						/>
					</div>

					{/* Model Upload */}
					<div
						style={{
							padding: "20px",
							backgroundColor: "#fff",
							borderRadius: "8px",
							border: "1px solid #e5e7eb",
							marginBottom: "20px",
						}}
					>
						<ModelUpload onUploadComplete={fetchModels} />
					</div>

					{/* ML Live Panel — ensemble gauge + per-model votes + toggles */}
					<MLLivePanel
						currentData={currentData}
						models={models}
						isCalibrating={isCalibrating}
						calibrationProgress={calibrationProgress}
						onToggle={toggleModel}
					/>
				</div>
			</div>
		</div>
	);
}

function MetricBox({
	label,
	value,
	tooltip,
}: {
	label: string;
	value: string;
	tooltip?: string;
}) {
	return (
		<div>
			<div style={{ fontSize: "12px", color: "#6b7280" }}>
				{tooltip ? <Tooltip text={tooltip}>{label}</Tooltip> : label}
			</div>
			<div style={{ fontSize: "18px", fontWeight: 600 }}>{value}</div>
		</div>
	);
}
