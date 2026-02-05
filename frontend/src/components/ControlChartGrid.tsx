import { useMemo } from 'react';
import { ControlChart } from './ControlChart';
import type { RawDataPoint, ChartDataPoint } from '../types';

// Define the data point type locally to avoid import issues
interface ControlChartDataPoint {
  time: number;
  value: number;
}

interface MethodThresholds {
  composite?: number;
  static?: number;
  slope?: number;
  ensemble?: number;
}

interface ControlChartGridProps {
  // Raw sensor data (columns from file)
  rawData: RawDataPoint[];
  selectedColumns: string[];

  // Method/computed data
  chartData: ChartDataPoint[];

  // Thresholds from backend (per method)
  methodThresholds?: MethodThresholds;

  // Default threshold for columns (can be overridden per column)
  defaultColumnThreshold?: number;
  columnThresholds?: Record<string, number>;

  // Configuration
  maxDataPoints?: number;
  chartHeight?: number;
}

// Detection methods to show
const METHODS = [
  { key: 'composite', label: 'Composite Score', useLog: true, defaultThreshold: 0.05 },
  { key: 'static', label: 'Static Score', useLog: true, defaultThreshold: 0.1 },
  { key: 'slope', label: 'Spectral Slope', useLog: false, defaultThreshold: 1.0 },
  { key: 'ensemble', label: 'Ensemble Probability', useLog: false, defaultThreshold: 0.7 },
] as const;

// Get unit for column based on name
const getColumnUnit = (column: string): string => {
  const lower = column.toLowerCase();
  if (lower.includes('temp')) return '°C';
  if (lower.includes('pressure') || lower.includes('pump')) return 'bar';
  if (lower.includes('flow')) return 'L/s';
  if (lower.includes('freq')) return 'Hz';
  return '';
};

// Shorten column name for display
const shortenName = (name: string): string => {
  let short = name
    .replace('(Mean)', '')
    .replace('(Arith. Mean)', '')
    .replace('TS ', '')
    .trim();

  if (short.length > 25) {
    short = short.substring(0, 22) + '...';
  }
  return short;
};

// Color palette for columns
const COLUMN_COLORS = [
  '#3b82f6', // Blue
  '#8b5cf6', // Purple
  '#22c55e', // Green
  '#f59e0b', // Orange
  '#ec4899', // Pink
  '#14b8a6', // Teal
  '#6366f1', // Indigo
  '#84cc16', // Lime
  '#ef4444', // Red
  '#06b6d4', // Cyan
];

// Calculate dynamic threshold (mean + 5*std) from data
function calculateDynamicThreshold(values: number[], multiplier: number = 5): number {
  if (values.length === 0) return 1;

  const validValues = values.filter((v) => !isNaN(v) && isFinite(v));
  if (validValues.length === 0) return 1;

  const mean = validValues.reduce((a, b) => a + b, 0) / validValues.length;
  const squaredDiffs = validValues.map((v) => Math.pow(v - mean, 2));
  const variance = squaredDiffs.reduce((a, b) => a + b, 0) / validValues.length;
  const std = Math.sqrt(variance);

  return mean + multiplier * std;
}

export function ControlChartGrid({
  rawData,
  selectedColumns,
  chartData,
  methodThresholds = {},
  defaultColumnThreshold,
  columnThresholds = {},
  maxDataPoints = 200000,
  chartHeight = 180,
}: ControlChartGridProps) {
  // Use all data (no limiting) - the store already handles the buffer size
  const limitedRawData = useMemo(() => {
    if (maxDataPoints && rawData.length > maxDataPoints) {
      return rawData.slice(-maxDataPoints);
    }
    return rawData;
  }, [rawData, maxDataPoints]);

  const limitedChartData = useMemo(() => {
    if (maxDataPoints && chartData.length > maxDataPoints) {
      return chartData.slice(-maxDataPoints);
    }
    return chartData;
  }, [chartData, maxDataPoints]);

  // Transform raw data for each column
  const columnChartData = useMemo(() => {
    const result: Record<string, ControlChartDataPoint[]> = {};

    selectedColumns.forEach((col) => {
      result[col] = limitedRawData
        .filter((point) => point[col] !== undefined && point[col] !== null)
        .map((point) => ({
          time: point.time,
          value: point[col] as number,
        }));
    });

    return result;
  }, [limitedRawData, selectedColumns]);

  // Transform method data
  const methodChartData = useMemo(() => {
    const result: Record<string, ControlChartDataPoint[]> = {};

    METHODS.forEach(({ key }) => {
      result[key] = limitedChartData
        .filter((point) => point[key] !== undefined && point[key] !== null)
        .map((point) => ({
          time: point.time,
          value: point[key] as number,
        }));
    });

    return result;
  }, [limitedChartData]);

  // Calculate thresholds for columns (5-sigma dynamic)
  const calculatedColumnThresholds = useMemo(() => {
    const result: Record<string, number> = {};

    selectedColumns.forEach((col) => {
      // Use provided threshold, or calculate dynamically
      if (columnThresholds[col] !== undefined) {
        result[col] = columnThresholds[col];
      } else if (defaultColumnThreshold !== undefined) {
        result[col] = defaultColumnThreshold;
      } else {
        // Calculate 5-sigma threshold from data
        const values = columnChartData[col]?.map((d) => d.value) || [];
        result[col] = calculateDynamicThreshold(values, 5);
      }
    });

    return result;
  }, [selectedColumns, columnChartData, columnThresholds, defaultColumnThreshold]);

  // Check if we have any method data
  const hasMethodData = METHODS.some(
    ({ key }) => methodChartData[key] && methodChartData[key].length > 0
  );

  // Debug: Log what data we have
  const lastPoint = limitedChartData[limitedChartData.length - 1];
  const lastPointKeys = lastPoint ? Object.keys(lastPoint) : [];
  console.log('ControlChartGrid Debug:', {
    chartDataLength: chartData.length,
    hasMethodData,
    methodDataLengths: METHODS.map(m => ({
      key: m.key,
      length: methodChartData[m.key]?.length || 0
    })),
    lastPointKeys,
    lastPointHasComposite: lastPoint?.composite !== undefined,
    lastPointValues: lastPoint ? {
      time: lastPoint.time,
      composite: lastPoint.composite,
      static: lastPoint.static,
      slope: lastPoint.slope,
      ensemble: lastPoint.ensemble,
    } : null,
  });

  return (
    <div>
      {/* Methods Section - Always show header, individual charts handle empty state */}
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ margin: '0 0 16px', fontSize: '18px', fontWeight: 500 }}>
          Detection Methods (Control Charts)
        </h2>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
            gap: '16px',
          }}
        >
          {METHODS.map(({ key, label, useLog, defaultThreshold }) => {
            const data = methodChartData[key] || [];

            // Get threshold: from props, or use default
            const threshold = methodThresholds[key as keyof MethodThresholds] ?? defaultThreshold;

            return (
              <ControlChart
                key={key}
                title={label}
                data={data}
                threshold={threshold}
                color="#8b5cf6"
                useLogScale={useLog}
                height={chartHeight}
              />
            );
          })}
        </div>
      </div>

      {/* Columns Section */}
      {selectedColumns.length > 0 && (
        <div>
          <h2 style={{ margin: '0 0 16px', fontSize: '18px', fontWeight: 500 }}>
            Sensor Data (5σ Control Charts)
          </h2>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '16px',
            }}
          >
            {selectedColumns.map((col, index) => {
              const data = columnChartData[col];
              if (!data || data.length === 0) return null;

              const threshold = calculatedColumnThresholds[col];

              return (
                <ControlChart
                  key={col}
                  title={shortenName(col)}
                  data={data}
                  threshold={threshold}
                  color={COLUMN_COLORS[index % COLUMN_COLORS.length]}
                  unit={getColumnUnit(col)}
                  useLogScale={false}
                  height={chartHeight}
                />
              );
            })}
          </div>
        </div>
      )}

    </div>
  );
}
