import { useMemo, useState } from "react";
import {
	CartesianGrid,
	Legend,
	Line,
	LineChart,
	ReferenceLine,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";

// Data point from the live stream
export interface CompositeDataPoint {
	time: number;
	composite_score: number;
	limit_threshold: number;
}

interface CompositeControlChartProps {
	data?: CompositeDataPoint;
	maxDataPoints?: number;
}

const COLORS = {
	normal: "#2196F3", // Blue - healthy signal
	breach: "#F44336", // Red - sigma breach (anomaly)
	threshold: "#F44336", // Red - threshold line
	grid: "#e5e7eb",
	axis: "#9ca3af",
};

// Buffer entry with breach status
interface BufferedDataPoint extends CompositeDataPoint {
	isBreach: boolean;
}

export function CompositeControlChart({
	data,
	maxDataPoints = 60,
}: CompositeControlChartProps) {
	const [buffer, setBuffer] = useState<BufferedDataPoint[]>([]);
	const [lastSeenTime, setLastSeenTime] = useState<number | null>(null);

	// Update buffer when new data arrives. Adjusting state during render is the
	// React-recommended pattern for accumulating prop-driven streams; the
	// `lastSeenTime` guard ensures we only setState when `data.time` has
	// actually changed, preventing render loops.
	if (data && lastSeenTime !== data.time) {
		setLastSeenTime(data.time);
		const isBreach = data.composite_score >= data.limit_threshold;

		setBuffer((prev) => {
			const newPoint: BufferedDataPoint = {
				...data,
				isBreach,
			};
			const updated = [...prev, newPoint];
			if (updated.length > maxDataPoints) {
				return updated.slice(-maxDataPoints);
			}
			return updated;
		});
	}

	// Calculate relative time labels for X-axis
	const chartData = useMemo(() => {
		if (buffer.length === 0) return [];

		const latestTime = buffer[buffer.length - 1].time;

		return buffer.map((point) => ({
			...point,
			relativeTime: point.time - latestTime, // Negative values going back in time
		}));
	}, [buffer]);

	// Check if current state is in breach
	const isCurrentlyBreaching =
		buffer.length > 0 && buffer[buffer.length - 1].isBreach;

	// Get the current threshold for the ReferenceLine
	const currentThreshold =
		buffer.length > 0 ? buffer[buffer.length - 1].limit_threshold : 0.05;

	// Custom tick formatter for relative time
	const formatXAxis = (value: number): string => {
		if (value === 0) return "Now";
		return `${Math.round(value)}s`;
	};

	// Custom Y-axis formatter for log scale
	const formatYAxis = (value: number): string => {
		if (value >= 1) return value.toFixed(1);
		if (value >= 0.1) return value.toFixed(2);
		if (value >= 0.01) return value.toFixed(3);
		return value.toExponential(1);
	};

	// Calculate safe domain for log scale (must avoid 0)
	const yDomain = useMemo((): [number, number] => {
		if (buffer.length === 0) return [0.001, 1];

		const scores = buffer.map((p) => p.composite_score).filter((v) => v > 0);
		const thresholds = buffer
			.map((p) => p.limit_threshold)
			.filter((v) => v > 0);
		const allValues = [...scores, ...thresholds];

		if (allValues.length === 0) return [0.001, 1];

		const minVal = Math.min(...allValues);
		const maxVal = Math.max(...allValues);

		// Add padding and ensure safe minimums for log scale
		const safeMin = Math.max(minVal * 0.5, 0.0001);
		const safeMax = maxVal * 2;

		return [safeMin, safeMax];
	}, [buffer]);

	// Empty state
	if (buffer.length === 0) {
		return (
			<div
				style={{
					padding: "20px",
					backgroundColor: "#fff",
					borderRadius: "8px",
					border: "1px solid #e5e7eb",
					boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
				}}
			>
				<h2 style={{ margin: "0 0 16px", fontSize: "18px" }}>
					Hydraulic Health Ratio (Composite)
				</h2>
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
			</div>
		);
	}

	return (
		<div
			style={{
				padding: "20px",
				backgroundColor: "#fff",
				borderRadius: "8px",
				border: isCurrentlyBreaching
					? "2px solid #F44336"
					: "1px solid #e5e7eb",
				boxShadow: isCurrentlyBreaching
					? "0 0 12px rgba(244, 67, 54, 0.3)"
					: "0 1px 3px rgba(0,0,0,0.1)",
				transition: "border-color 0.3s, box-shadow 0.3s",
			}}
		>
			{/* Header */}
			<div
				style={{
					display: "flex",
					justifyContent: "space-between",
					alignItems: "center",
					marginBottom: "16px",
				}}
			>
				<h2 style={{ margin: 0, fontSize: "18px" }}>
					Hydraulic Health Ratio (Composite)
				</h2>
				{isCurrentlyBreaching && (
					<span
						style={{
							padding: "4px 12px",
							borderRadius: "9999px",
							fontSize: "12px",
							fontWeight: 600,
							backgroundColor: "#fee2e2",
							color: "#991b1b",
							animation: "pulse 1.5s ease-in-out infinite",
						}}
					>
						SIGMA BREACH
					</span>
				)}
			</div>

			{/* Chart */}
			<div style={{ width: "100%", height: "300px" }}>
				<ResponsiveContainer>
					<LineChart
						data={chartData}
						margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
					>
						<CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} />

						{/* X-Axis: Relative time window */}
						<XAxis
							dataKey="relativeTime"
							tickFormatter={formatXAxis}
							stroke={COLORS.axis}
							fontSize={12}
							label={{
								value: "Time Window",
								position: "insideBottomRight",
								offset: -5,
								fontSize: 11,
								fill: COLORS.axis,
							}}
						/>

						{/* Y-Axis: Logarithmic scale */}
						<YAxis
							scale="log"
							domain={yDomain}
							tickFormatter={formatYAxis}
							stroke={COLORS.axis}
							fontSize={12}
							allowDataOverflow={false}
						/>

						<Tooltip
							formatter={(value: number, name: string) => {
								const formatted =
									value < 0.001 ? value.toExponential(3) : value.toFixed(4);
								return [
									formatted,
									name === "composite_score" ? "Composite Score" : "Threshold",
								];
							}}
							labelFormatter={(label: number) => {
								if (label === 0) return "Time: Now";
								return `Time: ${label.toFixed(1)}s`;
							}}
							contentStyle={{
								backgroundColor: "rgba(255, 255, 255, 0.95)",
								border: "1px solid #e5e7eb",
								borderRadius: "6px",
							}}
						/>

						<Legend
							formatter={(value) => {
								if (value === "composite_score")
									return "Composite Health Score";
								if (value === "limit_threshold") return "3-Sigma Limit";
								return value;
							}}
						/>

						{/* Threshold line - Red dashed reference */}
						<ReferenceLine
							y={currentThreshold}
							stroke={COLORS.threshold}
							strokeDasharray="8 4"
							strokeWidth={2}
							label={{
								value: `Limit: ${currentThreshold.toFixed(3)}`,
								position: "insideTopRight",
								fill: COLORS.threshold,
								fontSize: 11,
							}}
						/>

						{/* Main composite score line */}
						<Line
							type="monotone"
							dataKey="composite_score"
							name="composite_score"
							stroke={isCurrentlyBreaching ? COLORS.breach : COLORS.normal}
							strokeWidth={3}
							dot={false}
							isAnimationActive={false}
						/>

						{/* Optional: Also plot threshold as a line to show if it changes over time */}
						<Line
							type="stepAfter"
							dataKey="limit_threshold"
							name="limit_threshold"
							stroke={COLORS.threshold}
							strokeWidth={2}
							strokeDasharray="8 4"
							dot={false}
							isAnimationActive={false}
						/>
					</LineChart>
				</ResponsiveContainer>
			</div>

			{/* CSS for pulse animation */}
			<style>
				{`
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
          }
        `}
			</style>
		</div>
	);
}
