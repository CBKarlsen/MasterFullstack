// API and WebSocket data types

export interface ModelMetadata {
  name: string;
  version: string;
  author: string;
  description: string;
  model_type: 'sklearn' | 'pytorch' | 'tensorflow' | 'builtin' | 'custom';
  input_type: 'single' | 'sequence';
  input_features: string[];
  output_type: string;
  sequence_length?: number;
  framework_version?: string;
  enabled: boolean;
  weight?: number;
  prediction_count?: number;
  avg_inference_ms?: number;
  last_error?: string | null;
}

export interface ModelPrediction {
  probability: number;
  confidence: number;
  classification?: string;
  traffic_light?: string;
  spectral_slope?: number;
  buffer_status?: string;
  error?: string;
  [key: string]: unknown;
}

export interface SimulationData {
  type: 'data' | 'status' | 'columns';
  time?: number;
  flow?: number;
  pressure_drop?: number;
  static_score?: number;
  composite_score?: number;
  turbulence_score?: number;
  spectral_slope?: number;
  traffic_light?: string;
  light_msg?: string;
  ml_probability?: number;
  models?: Record<string, ModelPrediction>;
  ensemble_probability?: number | null;  // null during calibration
  message?: string;
  status?: string;
  calibration_progress?: number;
  raw?: Record<string, number>;  // Raw column data
  columns?: string[];  // Available column names
  limit_threshold?: number;
  static_threshold?: number;
  current_sigma?: number;
}

export interface ChartDataPoint {
  time: number;
  ensemble?: number | null;  // Can be null during calibration
  [key: string]: number | null | undefined;
}

export interface RawDataPoint {
  time: number;
  [key: string]: number;
}

export type TrafficLight = 'green' | 'yellow' | 'red' | 'gray';

export interface DataFile {
  name: string;
  size_bytes: number;
  size_human: string;
  modified: number;
  is_folder?: boolean;
  file_count?: number;
}

export interface SimulationConfig {
  action: 'start' | 'stop';
  file?: string;
  speed?: number;
}

export interface ColumnMetadata {
  name: string;
  type: string;
  unit: string;
}

export interface ColumnInfo {
  filename: string;
  columns: ColumnMetadata[];
}
