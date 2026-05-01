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
import type { ChartDataPoint, ModelMetadata } from "../types";

interface ProbabilityChartProps {
	data: ChartDataPoint[];
	models: ModelMetadata[];
	isCalibrating?: boolean;
	calibrationProgress?: number;
}

const MODEL_COLORS: Record<string, string> = {
	"FFT Physics": "#3b82f6",
	"Random Forest": "#22c55e",
	ensemble: "#8b5cf6",
	lstm: "#f59e0b",
	svm: "#ec4899",
	xgboost: "#14b8a6",
};

const getModelColor = (name: string, index: number): string => {
	if (MODEL_COLORS[name]) return MODEL_COLORS[name];
	const hue = (index * 137) % 360;
	return `hsl(${hue}, 70%, 50%)`;
};

export function ProbabilityChart({
	data,
	models,
	isCalibrating,
	calibrationProgress,
}: ProbabilityChartProps) {
	// ── calibrating state ────────────────────────────────────────────────────
	if (isCalibrating) {
		const progress = calibrationProgress
			? Math.round(calibrationProgress * 100)
			: 0;
		return (
			<div
				style={{
					height: "300px",
					display: "flex",
					flexDirection: "column",
					alignItems: "center",
					justifyContent: "center",
					color: "#6b7280",
					border: "1px dashed #e5e7eb",
					borderRadius: "8px",
					gap: "16px",
				}}
			>
				<div style={{ fontSize: "16px", fontWeight: 500 }}>
					Calibrating baseline (~20s) — do not stop the simulation
				</div>
				<div
					style={{
						width: "200px",
						height: "8px",
						backgroundColor: "#e5e7eb",
						borderRadius: "4px",
						overflow: "hidden",
					}}
				>
					<div
						style={{
							width: `${progress}%`,
							height: "100%",
							backgroundColor: "#3b82f6",
							transition: "width 0.3s ease",
						}}
					/>
				</div>
				<div style={{ fontSize: "14px", color: "#9ca3af" }}>
					{progress}% complete
				</div>
			</div>
		);
	}

	const hasValidData = data.some(
		(p) => p.ensemble !== undefined && p.ensemble !== null,
	);

	if (data.length === 0 || !hasValidData) {
		return (
			<div
				style={{
					height: "300px",
					display: "flex",
					alignItems: "center",
					justifyContent: "center",
					color: "#9ca3af",
					border: "1px dashed #e5e7eb",
					borderRadius: "8px",
				}}
			>
				Waiting for data...
			</div>
		);
	}

	const activeModelNames = models.filter((m) => m.enabled).map((m) => m.name);
	const validData = data.filter(
		(p) => p.ensemble !== undefined && p.ensemble !== null,
	);

	return (
		<div style={{ width: "100%", height: "300px" }}>
			<ResponsiveContainer>
				<ComposedChart
					data={validData}
					margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
				>
					<CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

					{/* ── Colored zone bands (drawn first so they sit behind lines) ── */}
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
						fontSize={12}
					/>
					<YAxis
						domain={[0, 1]}
						tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
						stroke="#9ca3af"
						fontSize={12}
					/>
					<Tooltip
						formatter={(value, name) => [
							value !== null && value !== undefined
								? `${(Number(value) * 100).toFixed(1)}%`
								: "N/A",
							name,
						]}
						labelFormatter={(label) => `Time: ${Number(label).toFixed(1)}s`}
					/>
					<Legend />

					{/* ── Ensemble — filled area ── */}
					<Area
						type="monotone"
						dataKey="ensemble"
						name="Ensemble"
						stroke={MODEL_COLORS.ensemble}
						strokeWidth={2.5}
						fill={MODEL_COLORS.ensemble}
						fillOpacity={0.15}
						dot={false}
						isAnimationActive={false}
						connectNulls={false}
					/>

					{/* ── Individual model lines (dashed so ensemble stands out) ── */}
					{activeModelNames.map((name, index) => (
						<Line
							key={name}
							type="monotone"
							dataKey={name}
							name={name}
							stroke={getModelColor(name, index)}
							strokeWidth={1.5}
							strokeDasharray="4 2"
							dot={false}
							isAnimationActive={false}
							connectNulls={false}
						/>
					))}
				</ComposedChart>
			</ResponsiveContainer>
		</div>
	);
}
