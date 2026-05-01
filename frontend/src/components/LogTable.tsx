import {
	createColumnHelper,
	flexRender,
	getCoreRowModel,
	getFilteredRowModel,
	getSortedRowModel,
	type SortingState,
	useReactTable,
} from "@tanstack/react-table";
import { useMemo, useRef, useState } from "react";
import type { LogEntry } from "../utils/resultLogStore";

// ── Constants ──────────────────────────────────────────────────────────────

const COL = createColumnHelper<LogEntry>();
const SORT_ICON: Record<string, string> = { asc: " ↑", desc: " ↓" };

// ── Actual-time inline editor ──────────────────────────────────────────────

interface ActualTimeCellProps {
	value: string | null;
	onChange: (v: string | null) => void;
}

function ActualTimeCell({ value, onChange }: ActualTimeCellProps) {
	const [editing, setEditing] = useState(false);
	const [draft, setDraft] = useState(value ?? "");

	if (!editing) {
		return (
			<span
				onClick={() => {
					setDraft(value ?? "");
					setEditing(true);
				}}
				title="Click to edit"
				style={{
					cursor: "pointer",
					color: value ? "#374151" : "#9ca3af",
					borderBottom: "1px dashed #d1d5db",
				}}
			>
				{value ?? "click to enter"}
			</span>
		);
	}

	const isValid = /^\d{2}:\d{2}:\d{2}$/.test(draft);
	return (
		<span style={{ display: "flex", gap: "4px", alignItems: "center" }}>
			<input
				autoFocus
				value={draft}
				onChange={(e) => setDraft(e.target.value)}
				placeholder="HH:MM:SS"
				style={{
					width: 80,
					padding: "2px 4px",
					fontFamily: "monospace",
					fontSize: "12px",
					border: `1px solid ${isValid || !draft ? "#e5e7eb" : "#fca5a5"}`,
					borderRadius: 4,
				}}
			/>
			<button
				onClick={() => {
					if (isValid) {
						onChange(draft);
						setEditing(false);
					} else if (!draft) {
						onChange(null);
						setEditing(false);
					}
				}}
				style={{
					fontSize: 11,
					padding: "2px 6px",
					borderRadius: 4,
					border: "none",
					background: "#2563eb",
					color: "#fff",
					cursor: "pointer",
				}}
			>
				✓
			</button>
			<button
				onClick={() => setEditing(false)}
				style={{
					fontSize: 11,
					padding: "2px 6px",
					borderRadius: 4,
					border: "1px solid #e5e7eb",
					background: "#fff",
					color: "#6b7280",
					cursor: "pointer",
				}}
			>
				✕
			</button>
		</span>
	);
}

// ── Column definitions ─────────────────────────────────────────────────────

function buildColumns(
	onUpdateActualTime: (e: LogEntry, v: string | null) => void,
) {
	return [
		COL.accessor("filename", {
			header: "File",
			cell: (i) => (
				<span
					title={i.getValue()}
					style={{
						maxWidth: 160,
						overflow: "hidden",
						textOverflow: "ellipsis",
						whiteSpace: "nowrap",
						display: "block",
					}}
				>
					{i.getValue()}
				</span>
			),
			size: 160,
		}),
		COL.accessor("sigma", { header: "σ", size: 40 }),
		COL.accessor("calibrationSeconds", { header: "Calib (s)", size: 72 }),
		COL.accessor("smoothingWindowSec", { header: "Window (s)", size: 80 }),
		COL.accessor("durationMin", {
			header: "Duration",
			cell: (i) => `${i.getValue()} min`,
			size: 80,
		}),
		COL.accessor("compositeCrossing", {
			header: "Composite onset",
			cell: (i) => i.getValue() ?? "—",
			size: 110,
		}),
		COL.accessor("staticCrossing", {
			header: "Static onset",
			cell: (i) => i.getValue() ?? "—",
			size: 100,
		}),
		COL.accessor("forecastEtaSigma3", {
			header: "ETA σ=3",
			cell: (i) => i.getValue() ?? "—",
			size: 80,
		}),
		COL.accessor("forecastEtaSigma4", {
			header: "ETA σ=4",
			cell: (i) => i.getValue() ?? "—",
			size: 80,
		}),
		COL.accessor("forecastEtaSigma5", {
			header: "ETA σ=5",
			cell: (i) => i.getValue() ?? "—",
			size: 80,
		}),
		COL.accessor("actualCloggingTime", {
			header: "Actual (editable)",
			cell: (i) => (
				<ActualTimeCell
					value={i.getValue()}
					onChange={(v) => onUpdateActualTime(i.row.original, v)}
				/>
			),
			size: 150,
		}),
		COL.accessor("bestFitModel", {
			header: "Best fit",
			cell: (i) => i.getValue() ?? "—",
			size: 90,
		}),
		COL.accessor("bestFitR2", {
			header: "R²",
			cell: (i) => (i.getValue() !== null ? i.getValue()!.toFixed(3) : "—"),
			size: 60,
		}),
		COL.accessor("peakMlProbability", {
			header: "Peak ML",
			cell: (i) =>
				i.getValue() !== null ? `${(i.getValue()! * 100).toFixed(1)}%` : "—",
			size: 70,
		}),
	];
}

// ── Main component ─────────────────────────────────────────────────────────

interface LogTableProps {
	entries: LogEntry[];
	onUpdateActualTime: (entry: LogEntry, value: string | null) => void;
	onImport: (csvText: string) => void;
}

export function LogTable({
	entries,
	onUpdateActualTime,
	onImport,
}: LogTableProps) {
	const [sorting, setSorting] = useState<SortingState>([]);
	const [globalFilter, setGlobalFilter] = useState("");
	const fileInputRef = useRef<HTMLInputElement>(null);

	const data = useMemo(() => [...entries].reverse(), [entries]);
	const columns = useMemo(
		() => buildColumns(onUpdateActualTime),
		[onUpdateActualTime],
	);

	const table = useReactTable({
		data,
		columns,
		state: { sorting, globalFilter },
		onSortingChange: setSorting,
		onGlobalFilterChange: setGlobalFilter,
		getCoreRowModel: getCoreRowModel(),
		getSortedRowModel: getSortedRowModel(),
		getFilteredRowModel: getFilteredRowModel(),
	});

	const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const file = e.target.files?.[0];
		if (!file) return;
		const reader = new FileReader();
		reader.onload = (ev) => {
			if (typeof ev.target?.result === "string") onImport(ev.target.result);
		};
		reader.readAsText(file);
		e.target.value = "";
	};

	return (
		<div>
			<div
				style={{
					display: "flex",
					alignItems: "center",
					gap: "10px",
					marginBottom: "10px",
				}}
			>
				<input
					value={globalFilter}
					onChange={(e) => setGlobalFilter(e.target.value)}
					placeholder="Filter by filename, model…"
					style={{
						flex: 1,
						padding: "6px 10px",
						borderRadius: "6px",
						border: "1px solid #e5e7eb",
						fontSize: "13px",
					}}
				/>
				<span
					style={{ fontSize: "12px", color: "#9ca3af", whiteSpace: "nowrap" }}
				>
					{table.getRowModel().rows.length} / {entries.length} rows
				</span>
				<input
					ref={fileInputRef}
					type="file"
					accept=".csv"
					style={{ display: "none" }}
					onChange={handleFileChange}
				/>
				<button
					onClick={() => fileInputRef.current?.click()}
					style={{
						padding: "6px 12px",
						borderRadius: "6px",
						border: "1px solid #e5e7eb",
						background: "#fff",
						color: "#374151",
						cursor: "pointer",
						fontSize: "12px",
						whiteSpace: "nowrap",
					}}
				>
					Import CSV
				</button>
			</div>

			<div style={{ overflowX: "auto" }}>
				<table
					style={{
						width: "100%",
						borderCollapse: "collapse",
						fontSize: "12px",
					}}
				>
					<thead>
						{table.getHeaderGroups().map((hg) => (
							<tr
								key={hg.id}
								style={{
									backgroundColor: "#f9fafb",
									borderBottom: "2px solid #e5e7eb",
								}}
							>
								{hg.headers.map((h) => (
									<th
										key={h.id}
										onClick={h.column.getToggleSortingHandler()}
										style={{
											padding: "8px 10px",
											textAlign: "left",
											fontWeight: 600,
											color: "#374151",
											whiteSpace: "nowrap",
											cursor: h.column.getCanSort() ? "pointer" : "default",
											userSelect: "none",
											width: h.getSize(),
										}}
									>
										{flexRender(h.column.columnDef.header, h.getContext())}
										{SORT_ICON[h.column.getIsSorted() as string] ?? ""}
									</th>
								))}
							</tr>
						))}
					</thead>
					<tbody>
						{table.getRowModel().rows.map((row, idx) => (
							<tr
								key={row.id}
								style={{
									backgroundColor: idx % 2 === 0 ? "#fff" : "#f9fafb",
									borderBottom: "1px solid #f3f4f6",
								}}
							>
								{row.getVisibleCells().map((cell) => (
									<td
										key={cell.id}
										style={{ padding: "7px 10px", color: "#374151" }}
									>
										{flexRender(cell.column.columnDef.cell, cell.getContext())}
									</td>
								))}
							</tr>
						))}
						{table.getRowModel().rows.length === 0 && (
							<tr>
								<td
									colSpan={columns.length}
									style={{
										padding: "24px",
										textAlign: "center",
										color: "#9ca3af",
									}}
								>
									No entries match the filter.
								</td>
							</tr>
						)}
					</tbody>
				</table>
			</div>
		</div>
	);
}
