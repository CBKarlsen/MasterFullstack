import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { RawDataPoint } from '../types';

interface RawDataChartProps {
  data: RawDataPoint[];
  selectedColumns: string[];
  height?: number;
}

// Color palette for data series
const COLORS = [
  '#3b82f6', // Blue
  '#ef4444', // Red
  '#22c55e', // Green
  '#f59e0b', // Orange
  '#8b5cf6', // Purple
  '#14b8a6', // Teal
  '#ec4899', // Pink
  '#6366f1', // Indigo
  '#84cc16', // Lime
  '#f97316', // Orange
];

// Get unit suffix based on column name
const getUnitSuffix = (column: string): string => {
  const lower = column.toLowerCase();
  if (lower.includes('temp')) return ' °C';
  if (lower.includes('pressure') || lower.includes('pump')) return ' bar';
  if (lower.includes('flow')) return ' L/s';
  return '';
};

// Shorten column name for display
const shortenColumnName = (name: string): string => {
  // Remove common suffixes
  let short = name
    .replace('(Mean)', '')
    .replace('(Arith. Mean)', '')
    .replace('TS ', '')
    .trim();

  // Truncate if still too long
  if (short.length > 20) {
    short = short.substring(0, 17) + '...';
  }

  return short;
};

export function RawDataChart({ data, selectedColumns, height = 300 }: RawDataChartProps) {
  if (data.length === 0) {
    return (
      <div
        style={{
          height: `${height}px`,
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

  if (selectedColumns.length === 0) {
    return (
      <div
        style={{
          height: `${height}px`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#9ca3af',
          border: '1px dashed #e5e7eb',
          borderRadius: '8px',
        }}
      >
        Select columns to visualize
      </div>
    );
  }

  // Calculate Y-axis domain based on selected columns
  const getColumnRange = (column: string): [number, number] => {
    let min = Infinity;
    let max = -Infinity;

    data.forEach((point) => {
      const value = point[column];
      if (value !== undefined && value !== null && !isNaN(value)) {
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
    });

    if (min === Infinity || max === -Infinity) {
      return [0, 1];
    }

    // Add some padding
    const padding = (max - min) * 0.1;
    return [min - padding, max + padding];
  };

  // Group columns by similar ranges for Y-axes
  // For simplicity, we'll use auto-scaling with a single Y-axis
  // In a more advanced version, we could use multiple Y-axes

  return (
    <div style={{ width: '100%', height: `${height}px` }}>
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
              maxHeight: '200px',
              overflowY: 'auto',
            }}
          />
          <Legend
            formatter={(value) => shortenColumnName(value)}
            wrapperStyle={{
              paddingTop: '10px',
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
