import { create } from 'zustand';
import type { SimulationData, ChartDataPoint, ModelMetadata, RawDataPoint } from '../types';

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
  setModels: (models: ModelMetadata[]) => void;
  toggleModel: (name: string, enabled: boolean) => void;
  updateModelWeight: (name: string, weight: number) => void;
  setSelectedColumns: (columns: string[]) => void;
  clearData: () => void;
}

export const useSimulationStore = create<SimulationState>((set, get) => ({
  currentData: null,
  status: 'Disconnected',
  calibrationProgress: 0,
  isCalibrating: false,
  chartData: [],
  rawChartData: [],
  maxDataPoints: 5000, // Keep last 200000 points (~10000 seconds at 20Hz)
  availableColumns: [],
  selectedColumns: [],
  models: [],

  updateData: (data: SimulationData) => {
    const state = get();

    // Handle status messages
    if (data.type === 'status') {
      set({
        status: data.message || 'Unknown',
        isCalibrating: data.message?.toLowerCase().includes('calibrat') || false,
      });
      return;
    }

    // Handle column metadata message
    if (data.type === 'columns' && data.columns) {
      // Auto-select some default columns if none selected
      const defaultColumns = data.columns.slice(0, 4);
      set({
        availableColumns: data.columns,
        selectedColumns: state.selectedColumns.length > 0 ? state.selectedColumns : defaultColumns,
      });
      return;
    }

    // Handle calibration progress - check if this data point is from calibration phase
    const isDataFromCalibration = data.status === 'calibrating';
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
      if (data.ensemble_probability !== null && data.ensemble_probability !== undefined) {
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
      }  if (data.limit_threshold !== undefined) {
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
        console.log('Store Debug - Incoming data sample:', {
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
      if (data.flow !== undefined) rawPoint['flow'] = data.flow;
      if (data.pressure_drop !== undefined) rawPoint['pressure_drop'] = data.pressure_drop;

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
        status: data.light_msg || (isDataFromCalibration ? 'Calibrating...' : 'Running'),
      });
    }
  },

  setModels: (models: ModelMetadata[]) => {
    set({ models });
  },

  toggleModel: (name: string, enabled: boolean) => {
    const state = get();
    const updatedModels = state.models.map((m) =>
      m.name === name ? { ...m, enabled } : m
    );
    set({ models: updatedModels });
  },

  updateModelWeight: (name: string, weight: number) => {
    const state = get();
    const updatedModels = state.models.map((m) =>
      m.name === name ? { ...m, weight } : m
    );
    set({ models: updatedModels });
  },

  setSelectedColumns: (columns: string[]) => {
    set({ selectedColumns: columns });
  },

  clearData: () => {
    set({
      currentData: null,
      chartData: [],
      rawChartData: [],
      status: 'Disconnected',
      calibrationProgress: 0,
      isCalibrating: false,
      availableColumns: [],
    });
  },
}));
