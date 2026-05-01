import { useMemo } from "react";

interface ColumnSelectorProps {
	availableColumns: string[];
	selectedColumns: string[];
	onSelectionChange: (columns: string[]) => void;
	disabled?: boolean;
}

// Group columns by category based on common patterns
const categorizeColumn = (name: string): string => {
	const lower = name.toLowerCase();
	if (lower.includes("temp")) return "Temperature";
	if (lower.includes("pressure") || lower.includes("pump")) return "Pressure";
	if (lower.includes("flow")) return "Flow";
	if (lower.includes("time") || lower.includes("date")) return "Time";
	return "Other";
};

// Color coding for categories
const CATEGORY_COLORS: Record<string, string> = {
	Temperature: "#ef4444",
	Pressure: "#3b82f6",
	Flow: "#22c55e",
	Time: "#6b7280",
	Other: "#8b5cf6",
};

export function ColumnSelector({
	availableColumns,
	selectedColumns,
	onSelectionChange,
	disabled = false,
}: ColumnSelectorProps) {
	// Group columns by category
	const groupedColumns = useMemo(() => {
		const groups: Record<string, string[]> = {};

		availableColumns.forEach((col) => {
			const category = categorizeColumn(col);
			if (!groups[category]) {
				groups[category] = [];
			}
			groups[category].push(col);
		});

		return groups;
	}, [availableColumns]);

	const handleToggle = (column: string) => {
		if (disabled) return;

		if (selectedColumns.includes(column)) {
			onSelectionChange(selectedColumns.filter((c) => c !== column));
		} else {
			onSelectionChange([...selectedColumns, column]);
		}
	};

	const handleSelectAll = (category: string) => {
		if (disabled) return;

		const categoryColumns = groupedColumns[category] || [];
		const allSelected = categoryColumns.every((col) =>
			selectedColumns.includes(col),
		);

		if (allSelected) {
			// Deselect all in category
			onSelectionChange(
				selectedColumns.filter((c) => !categoryColumns.includes(c)),
			);
		} else {
			// Select all in category
			const newSelection = new Set([...selectedColumns, ...categoryColumns]);
			onSelectionChange(Array.from(newSelection));
		}
	};

	if (availableColumns.length === 0) {
		return (
			<div
				style={{
					padding: "16px",
					backgroundColor: "#fff",
					borderRadius: "8px",
					border: "1px solid #e5e7eb",
				}}
			>
				<h3 style={{ margin: "0 0 12px", fontSize: "16px" }}>Data Columns</h3>
				<p style={{ color: "#6b7280", fontSize: "14px", margin: 0 }}>
					Start simulation to see available columns
				</p>
			</div>
		);
	}

	return (
		<div
			style={{
				padding: "16px",
				backgroundColor: "#fff",
				borderRadius: "8px",
				border: "1px solid #e5e7eb",
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
				<h3 style={{ margin: 0, fontSize: "16px" }}>
					Data Columns ({selectedColumns.length}/{availableColumns.length})
				</h3>
				<button
					onClick={() => onSelectionChange([])}
					disabled={disabled || selectedColumns.length === 0}
					style={{
						padding: "4px 8px",
						fontSize: "12px",
						border: "1px solid #e5e7eb",
						borderRadius: "4px",
						backgroundColor: "#fff",
						cursor: disabled ? "not-allowed" : "pointer",
						opacity: disabled || selectedColumns.length === 0 ? 0.5 : 1,
					}}
				>
					Clear All
				</button>
			</div>

			<div
				style={{
					maxHeight: "300px",
					overflowY: "auto",
					display: "flex",
					flexDirection: "column",
					gap: "12px",
				}}
			>
				{Object.entries(groupedColumns).map(([category, columns]) => {
					const allSelected = columns.every((col) =>
						selectedColumns.includes(col),
					);
					const someSelected = columns.some((col) =>
						selectedColumns.includes(col),
					);

					return (
						<div key={category}>
							{/* Category header */}
							<div
								style={{
									display: "flex",
									alignItems: "center",
									gap: "8px",
									marginBottom: "6px",
									cursor: disabled ? "not-allowed" : "pointer",
								}}
								onClick={() => handleSelectAll(category)}
							>
								<div
									style={{
										width: "12px",
										height: "12px",
										borderRadius: "3px",
										backgroundColor: CATEGORY_COLORS[category] || "#6b7280",
									}}
								/>
								<span
									style={{
										fontSize: "13px",
										fontWeight: 600,
										color: "#374151",
									}}
								>
									{category}
								</span>
								<input
									type="checkbox"
									checked={allSelected}
									ref={(el) => {
										if (el) el.indeterminate = someSelected && !allSelected;
									}}
									onChange={() => handleSelectAll(category)}
									disabled={disabled}
									style={{ marginLeft: "auto" }}
								/>
							</div>

							{/* Columns in category */}
							<div
								style={{
									display: "flex",
									flexDirection: "column",
									gap: "4px",
									paddingLeft: "20px",
								}}
							>
								{columns.map((col) => (
									<label
										key={col}
										style={{
											display: "flex",
											alignItems: "center",
											gap: "8px",
											fontSize: "13px",
											color: "#4b5563",
											cursor: disabled ? "not-allowed" : "pointer",
											opacity: disabled ? 0.6 : 1,
										}}
									>
										<input
											type="checkbox"
											checked={selectedColumns.includes(col)}
											onChange={() => handleToggle(col)}
											disabled={disabled}
										/>
										<span
											style={{
												overflow: "hidden",
												textOverflow: "ellipsis",
												whiteSpace: "nowrap",
											}}
											title={col}
										>
											{col}
										</span>
									</label>
								))}
							</div>
						</div>
					);
				})}
			</div>
		</div>
	);
}
