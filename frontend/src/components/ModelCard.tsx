import axios from "axios";
import type { ModelMetadata, ModelPrediction } from "../types";

interface ModelCardProps {
	model: ModelMetadata;
	prediction?: ModelPrediction;
	onToggle: (name: string, enabled: boolean) => void;
}

export function ModelCard({ model, prediction, onToggle }: ModelCardProps) {
	const handleToggle = async () => {
		try {
			await axios.put(
				`/api/models/${model.name}/enable?enabled=${!model.enabled}`,
			);
			onToggle(model.name, !model.enabled);
		} catch (error) {
			console.error("Failed to toggle model:", error);
		}
	};

	const probabilityColor = (prob: number) => {
		if (prob < 0.2) return "#22c55e";
		if (prob < 0.7) return "#eab308";
		return "#ef4444";
	};

	return (
		<div
			style={{
				border: "1px solid #e5e7eb",
				borderRadius: "8px",
				padding: "16px",
				backgroundColor: model.enabled ? "#fff" : "#f9fafb",
				opacity: model.enabled ? 1 : 0.6,
				transition: "all 0.2s",
			}}
		>
			<div
				style={{
					display: "flex",
					justifyContent: "space-between",
					alignItems: "start",
				}}
			>
				<div>
					<h3 style={{ margin: 0, fontSize: "16px", fontWeight: 600 }}>
						{model.name}
					</h3>
					<p style={{ margin: "4px 0 0", fontSize: "12px", color: "#6b7280" }}>
						{model.model_type} | {model.input_type}
						{model.input_type === "sequence" &&
							` (${model.sequence_length} steps)`}
					</p>
				</div>
				<label style={{ cursor: "pointer" }}>
					<input
						type="checkbox"
						checked={model.enabled}
						onChange={handleToggle}
						style={{ width: "20px", height: "20px" }}
					/>
				</label>
			</div>

			{model.description && (
				<p style={{ margin: "8px 0", fontSize: "13px", color: "#4b5563" }}>
					{model.description}
				</p>
			)}

			{prediction && model.enabled && (
				<div
					style={{
						marginTop: "12px",
						padding: "8px",
						backgroundColor: "#f3f4f6",
						borderRadius: "4px",
					}}
				>
					<div
						style={{
							display: "flex",
							justifyContent: "space-between",
							alignItems: "center",
						}}
					>
						<span style={{ fontSize: "13px", color: "#6b7280" }}>
							Probability:
						</span>
						<span
							style={{
								fontSize: "18px",
								fontWeight: "bold",
								color: probabilityColor(prediction.probability),
							}}
						>
							{(prediction.probability * 100).toFixed(1)}%
						</span>
					</div>
					{prediction.confidence !== undefined && (
						<div
							style={{ fontSize: "12px", color: "#9ca3af", marginTop: "4px" }}
						>
							Confidence: {(prediction.confidence * 100).toFixed(0)}%
						</div>
					)}
					{prediction.buffer_status && prediction.buffer_status !== "ready" && (
						<div
							style={{ fontSize: "12px", color: "#f59e0b", marginTop: "4px" }}
						>
							Buffer: {prediction.buffer_status}
						</div>
					)}
					{prediction.error && (
						<div
							style={{ fontSize: "12px", color: "#ef4444", marginTop: "4px" }}
						>
							Error: {prediction.error}
						</div>
					)}
				</div>
			)}

			<div style={{ marginTop: "8px", fontSize: "11px", color: "#9ca3af" }}>
				by {model.author} • v{model.version}
				{model.prediction_count !== undefined && model.prediction_count > 0 && (
					<> • {model.prediction_count} predictions</>
				)}
			</div>
		</div>
	);
}
