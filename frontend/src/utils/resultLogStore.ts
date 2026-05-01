/**
 * Accumulates batch analysis results in localStorage and exports as CSV.
 */

const LOG_STORAGE_KEY = "analysis_result_log";

export interface LogEntry {
	filename: string;
	timestamp: string;
	sigma: number;
	calibrationSeconds: number;
	smoothingWindowSec: number;
	durationMin: number;
	totalPoints: number;
	samplingHz: number;
	compositeThreshold: number;
	staticThreshold: number;
	compositeCrossing: string | null;
	staticCrossing: string | null;
	forecastOnset: string | null;
	forecastEtaSigma3: string | null;
	forecastEtaSigma4: string | null;
	forecastEtaSigma5: string | null;
	actualCloggingTime: string | null;
	bestFitModel: string | null;
	bestFitR2: number | null;
	peakMlProbability: number | null;
}

// ── Helpers ────────────────────────────────────────────────────────────────

export function hhmmssToSeconds(s: string): number | null {
	const parts = s.split(":").map(Number);
	if (parts.length !== 3 || parts.some(isNaN)) return null;
	return parts[0] * 3600 + parts[1] * 60 + parts[2];
}

// ── CRUD ───────────────────────────────────────────────────────────────────

export function loadLog(): LogEntry[] {
	try {
		const raw = localStorage.getItem(LOG_STORAGE_KEY);
		return raw ? (JSON.parse(raw) as LogEntry[]) : [];
	} catch {
		return [];
	}
}

function persistLog(log: LogEntry[]): void {
	localStorage.setItem(LOG_STORAGE_KEY, JSON.stringify(log));
}

export function appendEntry(entry: LogEntry): void {
	const log = loadLog();
	log.push(entry);
	persistLog(log);
}

export function updateEntryActualTime(
	timestamp: string,
	filename: string,
	value: string | null,
): void {
	const log = loadLog();
	const entry = log.find(
		(e) => e.timestamp === timestamp && e.filename === filename,
	);
	if (entry) {
		entry.actualCloggingTime = value;
		persistLog(log);
	}
}

export function clearLog(): void {
	localStorage.removeItem(LOG_STORAGE_KEY);
}

// ── CSV import ─────────────────────────────────────────────────────────────

function parseNum(s: string): number | null {
	const n = Number(s);
	return s === "" || isNaN(n) ? null : n;
}

function parseStr(s: string): string | null {
	return s === "" ? null : s;
}

export function importFromCsv(csvText: string): number {
	const lines = csvText.trim().split(/\r?\n/);
	if (lines.length < 2) return 0;

	const headers = lines[0].split(",");
	const idx = (key: string) => headers.indexOf(key);

	const existing = loadLog();
	const existingKeys = new Set(
		existing.map((e) => `${e.filename}__${e.timestamp}`),
	);

	let imported = 0;
	for (const line of lines.slice(1)) {
		if (!line.trim()) continue;
		const v = line.split(",");
		const get = (key: string) => (v[idx(key)] ?? "").trim();

		const entry: LogEntry = {
			filename: get("filename"),
			timestamp: get("timestamp"),
			sigma: Number(get("sigma")),
			calibrationSeconds: Number(get("calibrationSeconds")),
			smoothingWindowSec: Number(get("smoothingWindowSec")),
			durationMin: Number(get("durationMin")),
			totalPoints: Number(get("totalPoints")),
			samplingHz: Number(get("samplingHz")),
			compositeThreshold: Number(get("compositeThreshold")),
			staticThreshold: Number(get("staticThreshold")),
			compositeCrossing: parseStr(get("compositeCrossing")),
			staticCrossing: parseStr(get("staticCrossing")),
			forecastOnset: parseStr(get("forecastOnset")),
			forecastEtaSigma3: parseStr(get("forecastEtaSigma3")),
			forecastEtaSigma4: parseStr(get("forecastEtaSigma4")),
			forecastEtaSigma5: parseStr(get("forecastEtaSigma5")),
			actualCloggingTime: parseStr(get("actualCloggingTime")),
			bestFitModel: parseStr(get("bestFitModel")),
			bestFitR2: parseNum(get("bestFitR2")),
			peakMlProbability: parseNum(get("peakMlProbability")),
		};

		const key = `${entry.filename}__${entry.timestamp}`;
		if (!existingKeys.has(key) && entry.filename) {
			existing.push(entry);
			existingKeys.add(key);
			imported++;
		}
	}

	if (imported > 0) persistLog(existing);
	return imported;
}

// ── CSV export ─────────────────────────────────────────────────────────────

export function exportCsv(): void {
	const entries = loadLog();
	if (!entries.length) return;

	const headers = Object.keys(entries[0]) as (keyof LogEntry)[];
	const toCell = (val: unknown): string => {
		if (val === null || val === undefined) return "";
		const s = String(val);
		return s.includes(",") ? `"${s}"` : s;
	};

	const rows = entries.map((e) => headers.map((h) => toCell(e[h])).join(","));
	const csv = [headers.join(","), ...rows].join("\n");

	const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
	const url = URL.createObjectURL(blob);
	const link = document.createElement("a");
	link.href = url;
	link.download = `analysis_log_${new Date().toISOString().slice(0, 10)}.csv`;
	link.click();
	URL.revokeObjectURL(url);
}
