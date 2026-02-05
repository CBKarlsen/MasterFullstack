import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { ChartDataPoint, ModelMetadata } from '../types';

interface ProbabilityChartProps {
  data: ChartDataPoint[];
  models: ModelMetadata[];
  isCalibrating?: boolean;
  calibrationProgress?: number;
}

// Color palette for different models
const MODEL_COLORS: Record<string, string> = {
  fft_physics: '#3b82f6',    // Blue
  ensemble: '#8b5cf6',        // Purple
  random_forest: '#22c55e',   // Green
  lstm: '#f59e0b',            // Orange
  svm: '#ec4899',             // Pink
  xgboost: '#14b8a6',         // Teal
};

const getModelColor = (name: string, index: number): string => {
  if (MODEL_COLORS[name]) return MODEL_COLORS[name];
  // Generate color from index for unknown models
  const hue = (index * 137) % 360; // Golden angle for good distribution
  return `hsl(${hue}, 70%, 50%)`;
};

export function ProbabilityChart({ data, models, isCalibrating, calibrationProgress }: ProbabilityChartProps) {
  // Show calibration state
  if (isCalibrating) {
    const progress = calibrationProgress ? Math.round(calibrationProgress * 100) : 0;
    return (
      <div
        style={{
          height: '300px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#6b7280',
          border: '1px dashed #e5e7eb',
          borderRadius: '8px',
          gap: '16px',
        }}
      >
        <div style={{ fontSize: '16px', fontWeight: 500 }}>Calibrating System...</div>
        <div
          style={{
            width: '200px',
            height: '8px',
            backgroundColor: '#e5e7eb',
            borderRadius: '4px',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              width: `${progress}%`,
              height: '100%',
              backgroundColor: '#3b82f6',
              transition: 'width 0.3s ease',
            }}
          />
        </div>
        <div style={{ fontSize: '14px', color: '#9ca3af' }}>{progress}% complete</div>
      </div>
    );
  }

  // Check if we have valid probability data (not just time)
  const hasValidData = data.some((point) => point.ensemble !== undefined && point.ensemble !== null);

  if (data.length === 0 || !hasValidData) {
    return (
      <div
        style={{
          height: '300px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#9ca3af',
          border: '1px dashed #e5e7eb',
          borderRadius: '8px',
        }}
      >
        Waiting for data...
      </div>
    );
  }

  // Get active model names from data
  const activeModelNames = models
    .filter((m) => m.enabled)
    .map((m) => m.name);

  // Filter data to only include points with valid ensemble probability
  const validData = data.filter((point) => point.ensemble !== undefined && point.ensemble !== null);

  return (
    <div style={{ width: '100%', height: '300px' }}>
      <ResponsiveContainer>
        <LineChart
          data={validData}
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
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            stroke="#9ca3af"
            fontSize={12}
          />
          <Tooltip
            formatter={(value, name) => [
              value !== null && value !== undefined
                ? `${(Number(value) * 100).toFixed(1)}%`
                : 'N/A',
              name,
            ]}
            labelFormatter={(label) => `Time: ${Number(label).toFixed(1)}s`}
          />
          <Legend />

          {/* Warning threshold */}
          <ReferenceLine y={0.2} stroke="#eab308" strokeDasharray="5 5" />
          {/* Critical threshold */}
          <ReferenceLine y={0.7} stroke="#ef4444" strokeDasharray="5 5" />

          {/* Ensemble line (always shown) */}
          <Line
            type="monotone"
            dataKey="ensemble"
            name="Ensemble"
            stroke={MODEL_COLORS.ensemble}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            connectNulls={false}
          />

          {/* Individual model lines */}
          {activeModelNames.map((name, index) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              name={name}
              stroke={getModelColor(name, index)}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
