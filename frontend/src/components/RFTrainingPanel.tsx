/**
 * RFTrainingPanel — in-card training UI for the built-in Random Forest model.
 *
 * Renders inside the Random Forest model card. The Random Forest is the SUPERVISED
 * member of the ensemble: training expects a data file/folder covering both a healthy
 * period and a clogging event, and the backend auto-labels each moment as
 * healthy-vs-clogged. The user picks a file, a sigma threshold, a calibration window,
 * and RF hyperparameters (n_estimators, max_depth) before launching training.
 *
 * Backend endpoints: GET /api/data (file list), GET .../random_forest/training-info
 * (current source + stats), POST .../random_forest/train?... (start, fire-and-forget
 * into a backend thread), GET .../random_forest/training-progress (polled while
 * training), POST .../random_forest/reset (revert to the default synthetic model).
 *
 * Progress model: train() only kicks off a background thread; the panel then polls
 * the progress endpoint every 600 ms and stops the interval once the phase reaches
 * "complete" (capturing the result) or "error". Transient poll failures are ignored
 * so a single dropped request does not abort the run.
 */
import axios from "axios";
import { useEffect, useRef, useState } from "react";
import type { DataFile } from "../types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TrainingProgress {
	phase: string;
	percent: number;
	message: string;
	result?: TrainingResult | null;
	error?: string | null;
}

export interface TrainingResult {
	message: string;
	file_used: string;
	sigma: number;
	n_healthy: number;
	n_clogged: number;
	train_accuracy: number;
	test_accuracy: number;
	feature_importances: Record<string, number>;
}

interface TrainingInfo {
	source: string;
	is_user_trained: boolean;
	stats?: {
		n_healthy: number;
		n_clogged: number;
		train_accuracy: number;
		test_accuracy: number;
		n_estimators: number;
		max_depth: number | null;
		feature_importances: Record<string, number>;
	} | null;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Selectable sigma values; higher sigma = stricter clogged threshold = fewer alerts.
const SIGMA_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8];

// RF hyperparameter: number of trees, framed as a speed/accuracy trade-off.
const N_ESTIMATORS_OPTIONS = [
	{ label: "Fast (50 trees)", value: 50 },
	{ label: "Balanced (100 trees)", value: 100 },
	{ label: "Accurate (200 trees)", value: 200 },
	{ label: "Very accurate (300 trees)", value: 300 },
];

// RF hyperparameter: tree depth cap (null = unlimited), framed as model complexity.
const MAX_DEPTH_OPTIONS = [
	{ label: "Simple (depth 4)", value: 4 },
	{ label: "Balanced (depth 8)", value: 8 },
	{ label: "Complex (depth 12)", value: 12 },
	{ label: "Unlimited", value: null },
];

// Maps backend training-progress phase keys to user-facing status text.
const PHASE_LABELS: Record<string, string> = {
	starting: "Starting…",
	analyzing: "Analyzing data…",
	extracting: "Extracting features…",
	training: "Training model…",
	complete: "Complete",
	error: "Failed",
};

// Friendly names for the 4 input features, used in the feature-importance readout.
const FEATURE_LABELS: Record<string, string> = {
	static_score: "Pressure deviation",
	composite_score: "Frequency energy ratio",
	turbulence_score: "Turbulence level",
	spectral_slope: "Spectral slope",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface RFTrainingPanelProps {
	onTrainComplete: () => void;
}

export function RFTrainingPanel({ onTrainComplete }: RFTrainingPanelProps) {
	const [files, setFiles] = useState<DataFile[]>([]);
	const [selectedFile, setSelectedFile] = useState<string>("");
	const [sigma, setSigma] = useState(3);
	const [calibrationSeconds, setCalibrationSeconds] = useState(120);
	const [includeWarnings, setIncludeWarnings] = useState(false);
	const [nEstimators, setNEstimators] = useState(100);
	const [maxDepth, setMaxDepth] = useState<number | null>(8);
	const [training, setTraining] = useState(false);
	const [progress, setProgress] = useState<TrainingProgress | null>(null);
	const [result, setResult] = useState<TrainingResult | null>(null);
	const [trainingInfo, setTrainingInfo] = useState<TrainingInfo | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [resetting, setResetting] = useState(false);
	const [open, setOpen] = useState(false);
	// Holds the active progress-polling interval so it can be cleared on completion.
	const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

	// On mount, load the selectable data files and the model's current training source.
	useEffect(() => {
		axios
			.get<DataFile[]>("/api/data")
			.then((r) => setFiles(r.data))
			.catch(() => {});
		axios
			.get<TrainingInfo>("/api/models/random_forest/training-info")
			.then((r) => setTrainingInfo(r.data))
			.catch(() => {});
	}, []);

	// While a training run is active, poll the backend progress endpoint every 600 ms.
	// The run executes in a backend thread, so this is the only way to observe it.
	useEffect(() => {
		if (!training) return;

		pollRef.current = setInterval(async () => {
			try {
				const res = await axios.get<TrainingProgress>(
					"/api/models/random_forest/training-progress",
				);
				const state = res.data;
				setProgress(state);

				// Terminal "complete" phase: stop polling, store the result, and refresh
				// the training-info banner from the returned stats.
				if (state.phase === "complete" && state.result) {
					clearInterval(pollRef.current!);
					setResult(state.result);
					setTrainingInfo({
						source: `user data: ${selectedFile}`,
						is_user_trained: true,
						stats: state.result as TrainingResult & {
							feature_importances: Record<string, number>;
						},
					});
					setTraining(false);
					onTrainComplete();
				} else if (state.phase === "error") {
					// Terminal "error" phase: stop polling and surface the message.
					clearInterval(pollRef.current!);
					setError(state.error ?? "Training failed");
					setTraining(false);
				}
			} catch {
				/* transient poll failure — keep trying */
			}
		}, 600);

		return () => {
			if (pollRef.current) clearInterval(pollRef.current!);
		};
	}, [training, selectedFile, onTrainComplete]);

	// Starts a training run: flips into the training state (which arms the poller) and
	// POSTs the chosen file/sigma/hyperparameters. The POST returns immediately.
	const handleTrain = async () => {
		if (!selectedFile) return;
		setTraining(true);
		setError(null);
		setResult(null);
		setProgress({ phase: "starting", percent: 2, message: "Starting…" });

		try {
			const params = new URLSearchParams({
				file: selectedFile,
				sigma: String(sigma),
				calibration_seconds: String(calibrationSeconds),
				include_warnings: String(includeWarnings),
				n_estimators: String(nEstimators),
				// max_depth is omitted entirely when "Unlimited" (null) is selected.
				...(maxDepth !== null ? { max_depth: String(maxDepth) } : {}),
			});
			await axios.post(`/api/models/random_forest/train?${params}`);
		} catch (err: unknown) {
			const e = err as {
				response?: { data?: { detail?: string } };
				message?: string;
			};
			setError(
				e.response?.data?.detail ?? e.message ?? "Could not start training",
			);
			setTraining(false);
			setProgress(null);
		}
	};

	// Discards the user-trained model and restores the default synthetic-data model.
	const handleReset = async () => {
		setResetting(true);
		setError(null);
		try {
			await axios.post("/api/models/random_forest/reset");
			setResult(null);
			setTrainingInfo({
				source: "synthetic data",
				is_user_trained: false,
				stats: null,
			});
			onTrainComplete();
		} catch {
			setError("Reset failed");
		} finally {
			setResetting(false);
		}
	};

	// Formats a 0–1 ratio as a percentage string.
	const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

	return (
		<div
			style={{
				marginTop: "18px",
				borderTop: "1px solid #e5e7eb",
				paddingTop: "18px",
			}}
		>
			<div
				style={{
					display: "flex",
					justifyContent: "space-between",
					alignItems: "center",
					marginBottom: "12px",
				}}
			>
				<div>
					{/* Shows whether the model is on default synthetic data or user-trained. */}
					<span style={{ fontSize: "13px", fontWeight: 600, color: "#374151" }}>
						Training data:{" "}
					</span>
					<span
						style={{
							fontSize: "13px",
							color: trainingInfo?.is_user_trained ? "#16a34a" : "#6b7280",
							fontWeight: trainingInfo?.is_user_trained ? 600 : 400,
						}}
					>
						{trainingInfo?.source ?? "loading…"}
					</span>
				</div>
				<div style={{ display: "flex", gap: "8px" }}>
					{trainingInfo?.is_user_trained && (
						<button
							onClick={handleReset}
							disabled={resetting}
							style={{
								padding: "5px 12px",
								borderRadius: "6px",
								border: "1px solid #e5e7eb",
								background: "#fff",
								fontSize: "12px",
								cursor: resetting ? "wait" : "pointer",
								color: "#6b7280",
							}}
						>
							{resetting ? "Resetting…" : "Reset to default"}
						</button>
					)}
					<button
						onClick={() => setOpen((o) => !o)}
						style={{
							padding: "5px 14px",
							borderRadius: "6px",
							border: "none",
							background: open ? "#e0e7ff" : "#6366f1",
							color: open ? "#4338ca" : "#fff",
							fontSize: "12px",
							fontWeight: 600,
							cursor: "pointer",
						}}
					>
						{open ? "Close" : "Train on your data"}
					</button>
				</div>
			</div>

			{trainingInfo?.is_user_trained && trainingInfo.stats && !open && (
				<div
					style={{
						padding: "10px 14px",
						backgroundColor: "#f0fdf4",
						border: "1px solid #bbf7d0",
						borderRadius: "8px",
						marginBottom: "12px",
						fontSize: "12px",
					}}
				>
					<div style={{ display: "flex", gap: "24px", flexWrap: "wrap" }}>
						<span>
							✓ Trained on{" "}
							<strong>
								{(
									trainingInfo.stats.n_healthy + trainingInfo.stats.n_clogged
								).toLocaleString()}
							</strong>{" "}
							samples
						</span>
						<span>
							Healthy:{" "}
							<strong>{trainingInfo.stats.n_healthy.toLocaleString()}</strong>
						</span>
						<span>
							Clogged:{" "}
							<strong>{trainingInfo.stats.n_clogged.toLocaleString()}</strong>
						</span>
						<span>
							Hold-out accuracy:{" "}
							<strong
								style={{
									color:
										trainingInfo.stats.test_accuracy > 0.85
											? "#16a34a"
											: "#d97706",
								}}
							>
								{pct(trainingInfo.stats.test_accuracy)}
							</strong>
						</span>
					</div>
				</div>
			)}

			{open && (
				<div
					style={{
						padding: "18px 20px",
						backgroundColor: "#f9fafb",
						border: "1px solid #e5e7eb",
						borderRadius: "10px",
						display: "flex",
						flexDirection: "column",
						gap: "16px",
					}}
				>
					<p
						style={{
							margin: 0,
							fontSize: "13px",
							color: "#6b7280",
							lineHeight: 1.6,
						}}
					>
						Pick a <strong>file</strong> or <strong>folder</strong> that covers
						both a healthy period <em>and</em> a clogging event. The system
						classifies each moment automatically.
					</p>

					<div>
						<label
							style={{
								display: "block",
								fontSize: "13px",
								fontWeight: 600,
								marginBottom: "6px",
							}}
						>
							Training data
							<span
								style={{
									fontSize: "11px",
									fontWeight: 400,
									color: "#9ca3af",
									marginLeft: "6px",
								}}
							>
								file or folder
							</span>
						</label>
						<select
							value={selectedFile}
							onChange={(e) => setSelectedFile(e.target.value)}
							style={{
								width: "100%",
								padding: "8px 10px",
								borderRadius: "6px",
								border: "1px solid #d1d5db",
								fontSize: "13px",
								backgroundColor: "#fff",
							}}
						>
							<option value="">— Select a file or folder —</option>
							{files.filter((f) => f.is_folder).length > 0 && (
								<optgroup label="📁 Folders (all files combined)">
									{files
										.filter((f) => f.is_folder)
										.map((f) => (
											<option key={f.name} value={f.name}>
												{f.name} — {f.file_count ?? "?"} files, {f.size_human}
											</option>
										))}
								</optgroup>
							)}
							{files.filter((f) => !f.is_folder).length > 0 && (
								<optgroup label="📄 Individual files">
									{files
										.filter((f) => !f.is_folder)
										.map((f) => (
											<option key={f.name} value={f.name}>
												{f.name} ({f.size_human})
											</option>
										))}
								</optgroup>
							)}
						</select>
						{selectedFile &&
							files.find((f) => f.name === selectedFile)?.is_folder && (
								<div
									style={{
										marginTop: "6px",
										padding: "8px 10px",
										backgroundColor: "#eff6ff",
										border: "1px solid #bfdbfe",
										borderRadius: "6px",
										fontSize: "12px",
										color: "#1d4ed8",
									}}
								>
									All CSV/XLSX files inside this folder will be concatenated.
								</div>
							)}
					</div>

					<div
						style={{
							display: "grid",
							gridTemplateColumns: "1fr 1fr",
							gap: "16px",
						}}
					>
						<div>
							<label
								style={{
									display: "block",
									fontSize: "13px",
									fontWeight: 600,
									marginBottom: "6px",
								}}
							>
								Sensitivity (σ)
								<span
									style={{
										fontSize: "11px",
										fontWeight: 400,
										color: "#9ca3af",
										marginLeft: "6px",
									}}
								>
									higher = fewer alerts
								</span>
							</label>
							<div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
								{SIGMA_OPTIONS.map((s) => (
									<button
										key={s}
										onClick={() => setSigma(s)}
										style={{
											padding: "4px 10px",
											borderRadius: "5px",
											border: `1px solid ${s === sigma ? "#6366f1" : "#e5e7eb"}`,
											background: s === sigma ? "#6366f1" : "#fff",
											color: s === sigma ? "#fff" : "#374151",
											fontSize: "12px",
											cursor: "pointer",
											fontWeight: s === sigma ? 600 : 400,
										}}
									>
										{s}σ
									</button>
								))}
							</div>
						</div>
						<div>
							<label
								style={{
									display: "block",
									fontSize: "13px",
									fontWeight: 600,
									marginBottom: "6px",
								}}
							>
								Calibration window
								<span
									style={{
										fontSize: "11px",
										fontWeight: 400,
										color: "#9ca3af",
										marginLeft: "6px",
									}}
								>
									seconds of healthy baseline
								</span>
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
									width: "90px",
									padding: "6px 10px",
									borderRadius: "6px",
									border: "1px solid #d1d5db",
									fontSize: "13px",
								}}
							/>
							<span
								style={{
									fontSize: "12px",
									color: "#6b7280",
									marginLeft: "6px",
								}}
							>
								s
							</span>
						</div>
					</div>

					<div
						style={{
							display: "grid",
							gridTemplateColumns: "1fr 1fr",
							gap: "16px",
						}}
					>
						<div>
							<label
								style={{
									display: "block",
									fontSize: "13px",
									fontWeight: 600,
									marginBottom: "6px",
								}}
							>
								Speed vs. accuracy
							</label>
							<div
								style={{ display: "flex", flexDirection: "column", gap: "4px" }}
							>
								{N_ESTIMATORS_OPTIONS.map((o) => (
									<label
										key={o.value}
										style={{
											display: "flex",
											alignItems: "center",
											gap: "8px",
											fontSize: "12px",
											cursor: "pointer",
										}}
									>
										<input
											type="radio"
											name="n_estimators"
											checked={nEstimators === o.value}
											onChange={() => setNEstimators(o.value)}
										/>
										{o.label}
									</label>
								))}
							</div>
						</div>
						<div>
							<label
								style={{
									display: "block",
									fontSize: "13px",
									fontWeight: 600,
									marginBottom: "6px",
								}}
							>
								Model complexity
							</label>
							<div
								style={{ display: "flex", flexDirection: "column", gap: "4px" }}
							>
								{MAX_DEPTH_OPTIONS.map((o) => (
									<label
										key={String(o.value)}
										style={{
											display: "flex",
											alignItems: "center",
											gap: "8px",
											fontSize: "12px",
											cursor: "pointer",
										}}
									>
										<input
											type="radio"
											name="max_depth"
											checked={maxDepth === o.value}
											onChange={() => setMaxDepth(o.value)}
										/>
										{o.label}
									</label>
								))}
							</div>
						</div>
					</div>

					<label
						style={{
							display: "flex",
							alignItems: "center",
							gap: "10px",
							fontSize: "13px",
							cursor: "pointer",
						}}
					>
						{/* When set, warning-phase samples are labelled clogged, enlarging the
						    positive class at the cost of some label noise. */}
						<input
							type="checkbox"
							checked={includeWarnings}
							onChange={(e) => setIncludeWarnings(e.target.checked)}
							style={{ width: "16px", height: "16px" }}
						/>
						<span>
							<strong>Include warning-phase points as clogged</strong>
							<span style={{ color: "#6b7280", marginLeft: "6px" }}>
								— gives more training data but adds some noise
							</span>
						</span>
					</label>

					{error && (
						<div
							style={{
								padding: "10px 14px",
								backgroundColor: "#fef2f2",
								border: "1px solid #fca5a5",
								borderRadius: "8px",
								fontSize: "13px",
								color: "#dc2626",
							}}
						>
							{error}
						</div>
					)}

					{training && progress && (
						<div
							style={{
								padding: "14px 16px",
								backgroundColor: "#f5f3ff",
								border: "1px solid #c4b5fd",
								borderRadius: "10px",
							}}
						>
							<div
								style={{
									display: "flex",
									justifyContent: "space-between",
									alignItems: "center",
									marginBottom: "8px",
								}}
							>
								<span
									style={{
										fontSize: "13px",
										fontWeight: 600,
										color: "#4c1d95",
									}}
								>
									{PHASE_LABELS[progress.phase] ?? progress.phase}
								</span>
								<span
									style={{
										fontSize: "13px",
										fontWeight: 700,
										color: "#6d28d9",
										fontVariantNumeric: "tabular-nums",
									}}
								>
									{progress.percent}%
								</span>
							</div>
							<div
								style={{
									height: "8px",
									backgroundColor: "#ddd6fe",
									borderRadius: "4px",
									overflow: "hidden",
								}}
							>
								<div
									style={{
										height: "100%",
										borderRadius: "4px",
										backgroundColor: "#7c3aed",
										width: `${progress.percent}%`,
										transition: "width 0.5s ease",
									}}
								/>
							</div>
							<div
								style={{ fontSize: "11px", color: "#7c3aed", marginTop: "6px" }}
							>
								{progress.message}
							</div>
						</div>
					)}

					<div style={{ display: "flex", justifyContent: "flex-end" }}>
						<button
							onClick={handleTrain}
							disabled={!selectedFile || training}
							style={{
								padding: "10px 28px",
								borderRadius: "8px",
								border: "none",
								background: !selectedFile || training ? "#a5b4fc" : "#6366f1",
								color: "#fff",
								fontWeight: 700,
								fontSize: "14px",
								cursor: !selectedFile || training ? "not-allowed" : "pointer",
							}}
						>
							{training ? "Training…" : "Train model"}
						</button>
					</div>

					{result && (
						<div
							style={{
								padding: "16px",
								backgroundColor: "#f0fdf4",
								border: "1px solid #86efac",
								borderRadius: "10px",
							}}
						>
							<div
								style={{
									fontWeight: 700,
									fontSize: "14px",
									color: "#15803d",
									marginBottom: "12px",
								}}
							>
								✓ Training complete
							</div>
							<div
								style={{
									display: "grid",
									gridTemplateColumns: "repeat(4, 1fr)",
									gap: "12px",
									marginBottom: "12px",
								}}
							>
								{[
									{
										label: "Healthy samples",
										value: result.n_healthy.toLocaleString(),
									},
									{
										label: "Clogged samples",
										value: result.n_clogged.toLocaleString(),
									},
									{
										label: "Training accuracy",
										value: pct(result.train_accuracy),
									},
									{
										label: "Hold-out accuracy",
										value: pct(result.test_accuracy),
									},
								].map(({ label, value }) => (
									<div
										key={label}
										style={{
											padding: "10px",
											backgroundColor: "#fff",
											borderRadius: "8px",
											border: "1px solid #bbf7d0",
											textAlign: "center",
										}}
									>
										<div
											style={{
												fontSize: "11px",
												color: "#6b7280",
												marginBottom: "2px",
											}}
										>
											{label}
										</div>
										<div
											style={{
												fontSize: "16px",
												fontWeight: 700,
												color: "#15803d",
											}}
										>
											{value}
										</div>
									</div>
								))}
							</div>
							{/* Feature-importance bars, sorted descending: which of the 4 inputs
								    the trained Random Forest relies on most. */}
							<div>
								<div
									style={{
										fontSize: "12px",
										fontWeight: 600,
										color: "#374151",
										marginBottom: "6px",
									}}
								>
									What the model learned to look at:
								</div>
								<div
									style={{
										display: "flex",
										flexDirection: "column",
										gap: "5px",
									}}
								>
									{Object.entries(result.feature_importances)
										.sort(([, a], [, b]) => b - a)
										.map(([feature, importance]) => (
											<div
												key={feature}
												style={{
													display: "flex",
													alignItems: "center",
													gap: "8px",
												}}
											>
												<span
													style={{
														fontSize: "12px",
														color: "#6b7280",
														width: "160px",
														flexShrink: 0,
													}}
												>
													{FEATURE_LABELS[feature] ?? feature}
												</span>
												<div
													style={{
														flex: 1,
														height: "8px",
														backgroundColor: "#dcfce7",
														borderRadius: "4px",
														overflow: "hidden",
													}}
												>
													<div
														style={{
															width: `${importance * 100}%`,
															height: "100%",
															backgroundColor: "#16a34a",
															borderRadius: "4px",
														}}
													/>
												</div>
												<span
													style={{
														fontSize: "12px",
														color: "#374151",
														width: "36px",
														textAlign: "right",
													}}
												>
													{pct(importance)}
												</span>
											</div>
										))}
								</div>
							</div>
						</div>
					)}
				</div>
			)}
		</div>
	);
}
