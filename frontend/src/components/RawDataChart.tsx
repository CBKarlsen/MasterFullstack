import {
	CartesianGrid,
	Legend,
	Line,
	LineChart,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";
import type { RawDataPoint } from "../types";

interface RawDataChartProps {
	data: RawDataPoint[];
	selectedColumns: string[];
	height?: number;
}

// Color palette for data series
const COLORS = [
	"#3b82f6", // Blue
	"#ef4444", // Red
	"#22c55e", // Green
	"#f59e0b", // Orange
	"#8b5cf6", // Purple
	"#14b8a6", // Teal
	"#ec4899", // Pink
	"#6366f1", // Indigo
	"#84cc16", // Lime
	"#f97316", // Orange
];

// Get unit suffix based on column name
const getUnitSuffix = (column: string): string => {
	const lower = column.toLowerCase();
	if (lower.includes("temp")) return " °C";
	if (lower.includes("pressure") || lower.includes("pump")) return " bar";
	if (lower.includes("flow")) return " L/s";
	return "";
};

// Shorten column name for display
const shortenColumnName = (name: string): string => {
	// Remove common suffixes
	let short = name
		.replace("(Mean)", "")
		.replace("(Arith. Mean)", "")
		.replace("TS ", "")
		.trim();

	// Truncate if still too long
	if (short.length > 20) {
		short = short.substring(0, 17) + "...";
	}

	return short;
};

export function RawDataChart({
	data,
	selectedColumns,
	height = 300,
}: RawDataChartProps) {
	if (data.length === 0) {
		return (
			<div
				style={{
					height: `${height}px`,
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

	if (selectedColumns.length === 0) {
		return (
			<div
				style={{
					height: `${height}px`,
					display: "flex",
					alignItems: "center",
					justifyContent: "center",
					color: "#9ca3af",
					border: "1px dashed #e5e7eb",
					borderRadius: "8px",
				}}
			>
				Select columns to visualize
			</div>
		);
	}

	// Y-axis uses Recharts' auto-scaling across all rendered series.

	return (
		<div style={{ width: "100%", height: `${height}px` }}>
			<ResponsiveContainer>
				<LineChart
					data={data}
					margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
				>
					<CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
					<XAxis
						dataKey="time"
						tickFormatter={(v) => `${v.toFixed(0)}s`}
						stroke="#9ca3af"
						fontSize={12}
					/>
					<YAxis
						stroke="#9ca3af"
						fontSize={12}
						tickFormatter={(v) => {
							if (Math.abs(v) >= 1000) {
								return `${(v / 1000).toFixed(1)}k`;
							}
							return v.toFixed(2);
						}}
					/>
					<Tooltip
						formatter={(value: number, name: string) => {
							const unit = getUnitSuffix(name);
							return [`${value?.toFixed(4)}${unit}`, shortenColumnName(name)];
						}}
						labelFormatter={(label) => `Time: ${Number(label).toFixed(1)}s`}
						contentStyle={{
							maxHeight: "200px",
							overflowY: "auto",
						}}
					/>
					<Legend
						formatter={(value) => shortenColumnName(value)}
						wrapperStyle={{
							paddingTop: "10px",
						}}
					/>

					{selectedColumns.map((column, index) => (
						<Line
							key={column}
							type="monotone"
							dataKey={column}
							name={column}
							stroke={COLORS[index % COLORS.length]}
							strokeWidth={1.5}
							dot={false}
							isAnimationActive={false}
							connectNulls={true}
						/>
					))}
				</LineChart>
			</ResponsiveContainer>
		</div>
	);
}
