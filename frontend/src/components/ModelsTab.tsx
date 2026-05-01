import axios from "axios";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSimulationStore } from "../store/simulationStore";
import type { ModelMetadata } from "../types";
import { IFTrainingPanel } from "./IFTrainingPanel";
import { RFTrainingPanel } from "./RFTrainingPanel";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function weightLabel(w: number): string {
	if (w <= 0.1) return "Muted — not contributing";
	if (w <= 0.4) return "Low influence";
	if (w <= 0.8) return "Below normal";
	if (w <= 1.2) return "Normal influence";
	if (w <= 1.6) return "High influence";
	if (w <= 2.0) return "Dominant";
	return "Maximum influence";
}

function weightColor(w: number): string {
	if (w <= 0.1) return "#9ca3af";
	if (w <= 0.8) return "#3b82f6";
	if (w <= 1.2) return "#10b981";
	if (w <= 1.6) return "#f59e0b";
	return "#ef4444";
}

const TYPE_LABELS: Record<string, string> = {
	builtin: "Built-in",
	sklearn: "scikit-learn",
	pytorch: "PyTorch",
	tensorflow: "TensorFlow",
	custom: "Custom",
};

const TYPE_COLORS: Record<string, string> = {
	builtin: "#7c3aed",
	sklearn: "#0284c7",
	pytorch: "#ea580c",
	tensorflow: "#16a34a",
	custom: "#6b7280",
};

// ---------------------------------------------------------------------------
// Model management card
// ---------------------------------------------------------------------------

interface ModelCardProps {
	model: ModelMetadata;
	onToggle: (name: string, enabled: boolean) => Promise<void>;
	onWeightChange: (name: string, weight: number) => void;
	onTrainComplete: () => void;
}

function ModelManagementCard({
	model,
	onToggle,
	onWeightChange,
	onTrainComplete,
}: ModelCardProps) {
	const [toggling, setToggling] = useState(false);
	const weight = model.weight ?? 1.0;

	const handleToggle = async () => {
		setToggling(true);
		await onToggle(model.name, !model.enabled);
		setToggling(false);
	};

	return (
		<div
			style={{
				backgroundColor: "#fff",
				border: "1px solid #e5e7eb",
				borderRadius: "12px",
				padding: "20px 24px",
				opacity: model.enabled ? 1 : 0.55,
				transition: "opacity 0.2s, box-shadow 0.2s",
				boxShadow: model.enabled ? "0 1px 4px rgba(0,0,0,0.06)" : "none",
			}}
		>
			<div
				style={{
					display: "flex",
					justifyContent: "space-between",
					alignItems: "flex-start",
					marginBottom: "14px",
				}}
			>
				<div>
					<div
						style={{
							display: "flex",
							alignItems: "center",
							gap: "8px",
							marginBottom: "4px",
						}}
					>
						<h3 style={{ margin: 0, fontSize: "16px", fontWeight: 700 }}>
							{model.name}
						</h3>
						<span
							style={{
								fontSize: "11px",
								fontWeight: 600,
								padding: "2px 7px",
								borderRadius: "4px",
								backgroundColor: TYPE_COLORS[model.model_type] + "20",
								color: TYPE_COLORS[model.model_type],
							}}
						>
							{TYPE_LABELS[model.model_type] ?? model.model_type}
						</span>
					</div>
					{model.description && (
						<p
							style={{
								margin: 0,
								fontSize: "13px",
								color: "#6b7280",
								maxWidth: "480px",
								lineHeight: 1.5,
							}}
						>
							{model.description}
						</p>
					)}
				</div>

				<div
					style={{
						display: "flex",
						alignItems: "center",
						gap: "8px",
						flexShrink: 0,
					}}
				>
					<span
						style={{
							fontSize: "12px",
							color: model.enabled ? "#16a34a" : "#9ca3af",
							fontWeight: 500,
						}}
					>
						{model.enabled ? "Active" : "Off"}
					</span>
					<button
						onClick={handleToggle}
						disabled={toggling}
						title={model.enabled ? "Disable model" : "Enable model"}
						style={{
							width: "42px",
							height: "24px",
							borderRadius: "12px",
							border: "none",
							cursor: toggling ? "wait" : "pointer",
							backgroundColor: model.enabled ? "#16a34a" : "#d1d5db",
							position: "relative",
							transition: "background-color 0.2s",
							flexShrink: 0,
						}}
					>
						<span
							style={{
								position: "absolute",
								top: "3px",
								left: model.enabled ? "20px" : "3px",
								width: "18px",
								height: "18px",
								borderRadius: "50%",
								backgroundColor: "#fff",
								transition: "left 0.2s",
								boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
							}}
						/>
					</button>
				</div>
			</div>

			<div
				style={{
					display: "flex",
					gap: "24px",
					marginBottom: "18px",
					fontSize: "12px",
					color: "#6b7280",
				}}
			>
				{(model.prediction_count ?? 0) > 0 && (
					<span>
						<strong style={{ color: "#374151" }}>
							{model.prediction_count!.toLocaleString()}
						</strong>{" "}
						predictions
					</span>
				)}
				{(model.avg_inference_ms ?? 0) > 0 && (
					<span>
						Avg speed:{" "}
						<strong style={{ color: "#374151" }}>
							{model.avg_inference_ms!.toFixed(1)} ms
						</strong>
					</span>
				)}
				{model.input_type === "sequence" && model.sequence_length && (
					<span>
						Window:{" "}
						<strong style={{ color: "#374151" }}>
							{model.sequence_length} steps
						</strong>
					</span>
				)}
				<span>
					v{model.version} · {model.author}
				</span>
			</div>

			{model.last_error && (
				<div
					style={{
						marginBottom: "14px",
						padding: "8px 12px",
						backgroundColor: "#fef2f2",
						border: "1px solid #fca5a5",
						borderRadius: "6px",
						fontSize: "12px",
						color: "#dc2626",
					}}
				>
					⚠ Last error: {model.last_error}
				</div>
			)}

			<div>
				<div
					style={{
						display: "flex",
						justifyContent: "space-between",
						alignItems: "baseline",
						marginBottom: "6px",
					}}
				>
					<label
						style={{ fontSize: "13px", fontWeight: 600, color: "#374151" }}
					>
						Trust level
					</label>
					<span
						style={{
							fontSize: "12px",
							fontWeight: 600,
							color: weightColor(weight),
						}}
					>
						{weightLabel(weight)}
					</span>
				</div>
				<div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
					<span style={{ fontSize: "11px", color: "#9ca3af", width: "28px" }}>
						Low
					</span>
					<input
						type="range"
						min={0}
						max={2}
						step={0.05}
						value={weight}
						disabled={!model.enabled}
						onChange={(e) =>
							onWeightChange(model.name, parseFloat(e.target.value))
						}
						style={{
							flex: 1,
							accentColor: weightColor(weight),
							cursor: model.enabled ? "pointer" : "not-allowed",
						}}
					/>
					<span
						style={{
							fontSize: "11px",
							color: "#9ca3af",
							width: "28px",
							textAlign: "right",
						}}
					>
						High
					</span>
				</div>
				<p style={{ margin: "6px 0 0", fontSize: "11px", color: "#9ca3af" }}>
					Adjusts how much this model's opinion counts in the final clogging
					score.
				</p>
			</div>

			{model.name === "Random Forest" && (
				<RFTrainingPanel onTrainComplete={onTrainComplete} />
			)}
			{model.name === "Isolation Forest" && (
				<IFTrainingPanel onTrainComplete={onTrainComplete} />
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Main ModelsTab
// ---------------------------------------------------------------------------

export function ModelsTab() {
	const { models, setModels, toggleModel, updateModelWeight } =
		useSimulationStore();
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const debounceRefs = useRef<Record<string, ReturnType<typeof setTimeout>>>(
		{},
	);

	const fetchModels = useCallback(async () => {
		setLoading(true);
		setError(null);
		try {
			const res = await axios.get<ModelMetadata[]>("/api/models");
			setModels(res.data);
		} catch {
			setError("Could not load models. Is the backend running?");
		} finally {
			setLoading(false);
		}
	}, [setModels]);

	useEffect(() => {
		fetchModels();
		const interval = setInterval(fetchModels, 5000);
		return () => clearInterval(interval);
	}, [fetchModels]);

	const handleToggle = useCallback(
		async (name: string, enabled: boolean) => {
			try {
				await axios.put(
					`/api/models/${encodeURIComponent(name)}/enable?enabled=${enabled}`,
				);
				toggleModel(name, enabled);
			} catch {
				setError(`Failed to ${enabled ? "enable" : "disable"} "${name}".`);
			}
		},
		[toggleModel],
	);

	const handleWeightChange = useCallback(
		(name: string, weight: number) => {
			updateModelWeight(name, weight);
			clearTimeout(debounceRefs.current[name]);
			debounceRefs.current[name] = setTimeout(async () => {
				try {
					await axios.put(
						`/api/models/${encodeURIComponent(name)}/weight?weight=${weight}`,
					);
				} catch {
					setError(`Failed to update weight for "${name}".`);
				}
			}, 300);
		},
		[updateModelWeight],
	);

	const activeCount = models.filter((m) => m.enabled).length;

	return (
		<div style={{ maxWidth: "860px", margin: "0 auto" }}>
			<div
				style={{
					display: "flex",
					justifyContent: "space-between",
					alignItems: "flex-start",
					marginBottom: "24px",
				}}
			>
				<div>
					<h2 style={{ margin: "0 0 4px", fontSize: "20px", fontWeight: 700 }}>
						Model Management
					</h2>
					<p style={{ margin: 0, fontSize: "13px", color: "#6b7280" }}>
						{activeCount} of {models.length} model
						{models.length !== 1 ? "s" : ""} active · Use the Trust level slider
						to control each model's influence on the clogging score.
					</p>
				</div>
				<button
					onClick={fetchModels}
					disabled={loading}
					style={{
						padding: "7px 14px",
						borderRadius: "6px",
						border: "1px solid #e5e7eb",
						background: "#fff",
						fontSize: "13px",
						cursor: loading ? "wait" : "pointer",
						color: "#374151",
					}}
				>
					{loading ? "Refreshing…" : "Refresh"}
				</button>
			</div>

			{error && (
				<div
					style={{
						marginBottom: "16px",
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

			<div
				style={{
					marginBottom: "20px",
					padding: "14px 18px",
					backgroundColor: "#f0f9ff",
					border: "1px solid #bae6fd",
					borderRadius: "10px",
					fontSize: "13px",
					color: "#0369a1",
					lineHeight: 1.6,
				}}
			>
				<strong>How the overall score works:</strong> Each active model votes on
				whether the pipe is clogging. The final score is a weighted average of
				all votes.
			</div>

			{loading && models.length === 0 ? (
				<div style={{ textAlign: "center", padding: "60px", color: "#9ca3af" }}>
					Loading models…
				</div>
			) : models.length === 0 ? (
				<div style={{ textAlign: "center", padding: "60px", color: "#9ca3af" }}>
					No models loaded yet. Upload a model file or restart the backend.
				</div>
			) : (
				<div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
					{models.map((model) => (
						<ModelManagementCard
							key={model.name}
							model={model}
							onToggle={handleToggle}
							onWeightChange={handleWeightChange}
							onTrainComplete={fetchModels}
						/>
					))}
				</div>
			)}
		</div>
	);
}
