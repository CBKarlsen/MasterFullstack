/**
 * simulationStore — the Zustand store holding all live (and batch) streaming
 * state for the dashboard.
 *
 * It receives decoded `SimulationData` frames from the WebSocket hook and
 * fans them out into the shapes the charts consume: `currentData` (latest
 * frame for the status card / metrics), `chartData` (per-time detection
 * scores, thresholds, sigma, and ML probabilities), and `rawChartData`
 * (raw sensor columns plus computed flow / pressure-drop). Both chart arrays
 * are kept as rolling windows trimmed to `maxDataPoints` so memory stays
 * bounded over long runs.
 *
 * Frames are polymorphic: a `type` of "status" or "columns" carries
 * connection/metadata rather than a sample, and samples themselves are tagged
 * `status: "calibrating"` while the backend builds its healthy baseline — the
 * store routes these to status/calibration flags and only emits chart points
 * for real, post-calibration samples (probabilities/scores stay absent until
 * then). `updateData` handles one frame; `updateDataBatch` applies a whole
 * animation-frame batch in a single `set` (the hook's preferred path) by
 * folding all frames into one merged update to avoid per-message store writes.
 */
import { create } from "zustand";
import type {
	ChartDataPoint,
	ModelMetadata,
	RawDataPoint,
	SimulationData,
} from "../types";

interface SimulationState {
	// Current values
	currentData: SimulationData | null;
	status: string;
	calibrationProgress: number;
	isCalibrating: boolean;

	// Historical data for charts (limited buffer)
	chartData: ChartDataPoint[];
	rawChartData: RawDataPoint[];
	maxDataPoints: number;

	// Available columns from the data file
	availableColumns: string[];
	selectedColumns: string[];

	// Models
	models: ModelMetadata[];

	// Actions
	updateData: (data: SimulationData) => void;
	updateDataBatch: (batch: SimulationData[]) => void;
	setModels: (models: ModelMetadata[]) => void;
	toggleModel: (name: string, enabled: boolean) => void;
	updateModelWeight: (name: string, weight: number) => void;
	setSelectedColumns: (columns: string[]) => void;
	clearData: () => void;
}

// Keep only the most recent `max` elements, preserving the rolling-window
// invariant for chart buffers.
function trimToMax<T>(arr: T[], max: number): T[] {
	if (arr.length <= max) return arr;
	return arr.slice(arr.length - max);
}

export const useSimulationStore = create<SimulationState>((set, get) => ({
	currentData: null,
	status: "Disconnected",
	calibrationProgress: 0,
	isCalibrating: false,
	chartData: [],
	rawChartData: [],
	maxDataPoints: 2000,
	availableColumns: [],
	selectedColumns: [],
	models: [],

	// Single-frame path. Charts in this app are driven via updateDataBatch;
	// this variant applies one SimulationData frame at a time with the same
	// status/columns/calibration routing.
	updateData: (data: SimulationData) => {
		const state = get();

		// Handle status messages
		if (data.type === "status") {
			set({
				status: data.message || "Unknown",
				isCalibrating:
					data.message?.toLowerCase().includes("calibrat") || false,
			});
			return;
		}

		// Handle column metadata message
		if (data.type === "columns" && data.columns) {
			// Auto-select some default columns if none selected
			const defaultColumns = data.columns.slice(0, 4);
			set({
				availableColumns: data.columns,
				selectedColumns:
					state.selectedColumns.length > 0
						? state.selectedColumns
						: defaultColumns,
			});
			return;
		}

		// Handle calibration progress - check if this data point is from calibration phase.
		// While calibrating, surface a percent in `status` and hold the flag set;
		// the first non-calibration sample clears it (detection results are valid from there on).
		const isDataFromCalibration = data.status === "calibrating";
		if (isDataFromCalibration && data.calibration_progress) {
			set({
				status: `Calibrating: ${Math.round(data.calibration_progress * 100)}%`,
				calibrationProgress: data.calibration_progress,
				isCalibrating: true,
			});
		} else if (!isDataFromCalibration) {
			set({ isCalibrating: false });
		}

		// Create chart data point
		if (data.time !== undefined) {
			// Probability chart point - only add ensemble if we have valid data
			const point: ChartDataPoint = {
				time: data.time,
			};

			// Only add ensemble probability if it's not null (i.e., after calibration)
			if (
				data.ensemble_probability !== null &&
				data.ensemble_probability !== undefined
			) {
				point.ensemble = data.ensemble_probability;
			}

			// Add each model's probability (only if we have model data)
			if (data.models && Object.keys(data.models).length > 0) {
				Object.entries(data.models).forEach(([name, pred]) => {
					point[name] = pred.probability || 0;
				});
			}

			// Add method fields - these are available after calibration
			// The backend sends these fields with actual values once calibration is complete
			if (data.static_score !== undefined && data.static_score !== null) {
				point.static = data.static_score;
			}
			if (data.composite_score !== undefined && data.composite_score !== null) {
				point.composite = data.composite_score;
			}
			if (data.spectral_slope !== undefined && data.spectral_slope !== null) {
				point.slope = data.spectral_slope;
			}
			if (data.limit_threshold !== undefined) {
				point.limit_threshold = data.limit_threshold;
			}
			if (data.static_threshold !== undefined) {
				point.static_threshold = data.static_threshold;
			}
			if (data.current_sigma !== undefined) {
				point.current_sigma = data.current_sigma;
			}

			// Debug: Log every 100th point to see what data we have
			if (state.chartData.length % 100 === 0) {
				console.log("Store Debug - Incoming data sample:", {
					time: data.time,
					status: data.status,
					hasComposite: data.composite_score !== undefined,
					compositeValue: data.composite_score,
					hasStatic: data.static_score !== undefined,
					staticValue: data.static_score,
					pointKeys: Object.keys(point),
				});
			}

			// Create raw data point for physics chart
			const rawPoint: RawDataPoint = {
				time: data.time,
			};

			// Add all raw column data
			if (data.raw) {
				Object.entries(data.raw).forEach(([col, value]) => {
					rawPoint[col] = value;
				});
			}

			// Also add computed values
			if (data.flow !== undefined) rawPoint["flow"] = data.flow;
			if (data.pressure_drop !== undefined)
				rawPoint["pressure_drop"] = data.pressure_drop;

			// Update chart data with rolling window.
			// The WebSocket hook batches messages via requestAnimationFrame,
			// so this runs at most ~60 times/sec — the spread copy is acceptable.
			const newChartData = [...state.chartData, point];
			if (newChartData.length > state.maxDataPoints) {
				newChartData.splice(0, newChartData.length - state.maxDataPoints);
			}

			const newRawChartData = [...state.rawChartData, rawPoint];
			if (newRawChartData.length > state.maxDataPoints) {
				newRawChartData.splice(0, newRawChartData.length - state.maxDataPoints);
			}

			set({
				currentData: data,
				chartData: newChartData,
				rawChartData: newRawChartData,
				status:
					data.light_msg ||
					(isDataFromCalibration ? "Calibrating..." : "Running"),
			});
		}
	},

	// Batched path (the one the WebSocket hook actually calls). Folds an entire
	// animation-frame's worth of frames into local accumulators, then commits a
	// single `set` — one re-render per frame regardless of how many samples
	// arrived. Mirrors updateData's per-frame routing exactly.
	updateDataBatch: (batch: SimulationData[]) => {
		if (batch.length === 0) return;
		const state = get();

		// Accumulators seeded from current state; only "status"/"columns"/calibration
		// frames mutate the metadata fields, while real samples append chart points.
		let nextStatus = state.status;
		let nextCalibrationProgress = state.calibrationProgress;
		let nextIsCalibrating = state.isCalibrating;
		let nextAvailableColumns = state.availableColumns;
		let nextSelectedColumns = state.selectedColumns;
		let lastDataMessage: SimulationData | null = null;

		const appendedChart: ChartDataPoint[] = [];
		const appendedRaw: RawDataPoint[] = [];

		for (const data of batch) {
			if (data.type === "status") {
				nextStatus = data.message || "Unknown";
				nextIsCalibrating =
					data.message?.toLowerCase().includes("calibrat") || false;
				continue;
			}

			if (data.type === "columns" && data.columns) {
				nextAvailableColumns = data.columns;
				if (nextSelectedColumns.length === 0) {
					nextSelectedColumns = data.columns.slice(0, 4);
				}
				continue;
			}

			const isFromCalibration = data.status === "calibrating";
			if (isFromCalibration && data.calibration_progress) {
				nextStatus = `Calibrating: ${Math.round(data.calibration_progress * 100)}%`;
				nextCalibrationProgress = data.calibration_progress;
				nextIsCalibrating = true;
			} else if (!isFromCalibration) {
				nextIsCalibrating = false;
			}

			if (data.time === undefined) continue;

			const point: ChartDataPoint = { time: data.time };
			if (
				data.ensemble_probability !== null &&
				data.ensemble_probability !== undefined
			) {
				point.ensemble = data.ensemble_probability;
			}
			if (data.models && Object.keys(data.models).length > 0) {
				Object.entries(data.models).forEach(([name, pred]) => {
					point[name] = pred.probability || 0;
				});
			}
			if (data.static_score !== undefined && data.static_score !== null) {
				point.static = data.static_score;
			}
			if (data.composite_score !== undefined && data.composite_score !== null) {
				point.composite = data.composite_score;
			}
			if (data.spectral_slope !== undefined && data.spectral_slope !== null) {
				point.slope = data.spectral_slope;
			}
			if (data.limit_threshold !== undefined) {
				point.limit_threshold = data.limit_threshold;
			}
			if (data.static_threshold !== undefined) {
				point.static_threshold = data.static_threshold;
			}
			if (data.current_sigma !== undefined) {
				point.current_sigma = data.current_sigma;
			}

			const rawPoint: RawDataPoint = { time: data.time };
			if (data.raw) {
				Object.entries(data.raw).forEach(([col, value]) => {
					rawPoint[col] = value;
				});
			}
			if (data.flow !== undefined) rawPoint["flow"] = data.flow;
			if (data.pressure_drop !== undefined)
				rawPoint["pressure_drop"] = data.pressure_drop;

			appendedChart.push(point);
			appendedRaw.push(rawPoint);
			lastDataMessage = data;

			nextStatus =
				data.light_msg ||
				(isFromCalibration ? "Calibrating..." : "Running");
		}

		// Append the batch's new points in one shot, then trim to the rolling
		// window. Reuse the existing array reference when nothing was appended
		// to avoid a needless re-render of chart consumers.
		const max = state.maxDataPoints;
		const mergedChart =
			appendedChart.length === 0
				? state.chartData
				: trimToMax([...state.chartData, ...appendedChart], max);
		const mergedRaw =
			appendedRaw.length === 0
				? state.rawChartData
				: trimToMax([...state.rawChartData, ...appendedRaw], max);

		set({
			status: nextStatus,
			calibrationProgress: nextCalibrationProgress,
			isCalibrating: nextIsCalibrating,
			availableColumns: nextAvailableColumns,
			selectedColumns: nextSelectedColumns,
			chartData: mergedChart,
			rawChartData: mergedRaw,
			currentData: lastDataMessage ?? state.currentData,
		});
	},

	setModels: (models: ModelMetadata[]) => {
		set({ models });
	},

	toggleModel: (name: string, enabled: boolean) => {
		const state = get();
		const updatedModels = state.models.map((m) =>
			m.name === name ? { ...m, enabled } : m,
		);
		set({ models: updatedModels });
	},

	updateModelWeight: (name: string, weight: number) => {
		const state = get();
		const updatedModels = state.models.map((m) =>
			m.name === name ? { ...m, weight } : m,
		);
		set({ models: updatedModels });
	},

	setSelectedColumns: (columns: string[]) => {
		set({ selectedColumns: columns });
	},

	// Reset all streamed state for a fresh run (called before starting a new
	// simulation); leaves loaded `models` untouched.
	clearData: () => {
		set({
			currentData: null,
			chartData: [],
			rawChartData: [],
			status: "Disconnected",
			calibrationProgress: 0,
			isCalibrating: false,
			availableColumns: [],
		});
	},
}));
