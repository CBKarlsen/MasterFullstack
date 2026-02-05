import { useState, useEffect, useRef, useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

// Data point from the live stream
export interface CompositeDataPoint {
  time: number;
  composite_score: number;
  limit_threshold: number;
}

interface CompositeChartProps {
  data?: CompositeDataPoint;
  maxDataPoints?: number;
}

const COLORS = {
  normal: '#2196F3',        // Blue - healthy signal
  normalFill: 'rgba(33, 150, 243, 0.3)',
  breach: '#F44336',        // Red - sigma breach
  breachFill: 'rgba(244, 67, 54, 0.4)',
  threshold: '#F44336',     // Red - threshold line
  grid: '#e5e7eb',
  axis: '#9ca3af',
};

// Buffer entry with breach status
interface BufferedDataPoint extends CompositeDataPoint {
  isBreach: boolean;
  // Safe value for log scale (replaces 0 with minimum)
  safeScore: number;
}

// Minimum value for log scale safety
const LOG_SAFE_MIN = 0.0001;

export function CompositeChart({
  data,
  maxDataPoints = 100,
}: CompositeChartProps) {
  const [buffer, setBuffer] = useState<BufferedDataPoint[]>([]);
  const latestTimeRef = useRef<number | null>(null);

  // Update buffer when new data arrives
  useEffect(() => {
    if (!data) return;

    // Avoid duplicate entries for the same time
    if (latestTimeRef.current === data.time) return;
    latestTimeRef.current = data.time;

    // Handle zero/negative values for log scale
    const safeScore = data.composite_score > 0 ? data.composite_score : LOG_SAFE_MIN;
    const isBreach = data.composite_score > data.limit_threshold;

    setBuffer((prev) => {
      const newPoint: BufferedDataPoint = {
        ...data,
        safeScore,
        isBreach,
      };
      const updated = [...prev, newPoint];
      // Maintain sliding window - remove oldest
      if (updated.length > maxDataPoints) {
        return updated.slice(-maxDataPoints);
      }
      return updated;
    });
  }, [data, maxDataPoints]);

  // Calculate relative time for display
  const chartData = useMemo(() => {
    if (buffer.length === 0) return [];

    const latestTime = buffer[buffer.length - 1].time;

    return buffer.map((point) => ({
      ...point,
      relativeTime: point.time - latestTime,
    }));
  }, [buffer]);

  // Check if currently breaching
  const isCurrentlyBreaching = buffer.length > 0 && buffer[buffer.length - 1].isBreach;

  // Get current threshold for reference line
  const currentThreshold = buffer.length > 0
    ? buffer[buffer.length - 1].limit_threshold
    : 0.05;

  // Format X-axis ticks to show relative time
  const formatXAxis = (value: number): string => {
    if (value === 0) return 'Now';
    return `${Math.round(value)}s`;
  };

  // Format Y-axis for log scale readability
  const formatYAxis = (value: number): string => {
    if (value >= 1) return value.toFixed(1);
    if (value >= 0.1) return value.toFixed(2);
    if (value >= 0.01) return value.toFixed(3);
    if (value >= 0.001) return value.toFixed(4);
    return value.toExponential(0);
  };

  // Calculate Y-axis domain for log scale
  const yDomain = useMemo((): [number, number] => {
    if (buffer.length === 0) return [LOG_SAFE_MIN, 1];

    const scores = buffer.map((p) => p.safeScore);
    const thresholds = buffer.map((p) => p.limit_threshold).filter((v) => v > 0);
    const allValues = [...scores, ...thresholds];

    const maxVal = Math.max(...allValues);

    // Use fixed minimum for log safety, dynamic maximum with padding
    return [LOG_SAFE_MIN, Math.max(maxVal * 2, 1)];
  }, [buffer]);

  // Dynamic colors based on breach state
  const areaStroke = isCurrentlyBreaching ? COLORS.breach : COLORS.normal;
  const areaFill = isCurrentlyBreaching ? COLORS.breachFill : COLORS.normalFill;

  // Empty state
  if (buffer.length === 0) {
    return (
      <div
        style={{
          padding: '20px',
          backgroundColor: '#fff',
          borderRadius: '8px',
          border: '1px solid #e5e7eb',
          boxShadow: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.06)',
        }}
      >
        <h2 style={{ margin: '0 0 16px', fontSize: '18px', fontWeight: 500 }}>
          Composite Health Index (Log Scale)
        </h2>
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
      </div>
    );
  }

  return (
    <div
      style={{
        padding: '20px',
        backgroundColor: '#fff',
        borderRadius: '8px',
        border: isCurrentlyBreaching ? '2px solid #F44336' : '1px solid #e5e7eb',
        boxShadow: isCurrentlyBreaching
          ? '0 4px 12px rgba(244, 67, 54, 0.25)'
          : '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.06)',
        transition: 'all 0.3s ease',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px',
        }}
      >
        <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 500 }}>
          Composite Health Index (Log Scale)
        </h2>
        {isCurrentlyBreaching && (
          <span
            style={{
              padding: '4px 12px',
              borderRadius: '4px',
              fontSize: '12px',
              fontWeight: 600,
              backgroundColor: '#F44336',
              color: '#fff',
              animation: 'pulse 1s ease-in-out infinite',
            }}
          >
            SIGMA BREACH
          </span>
        )}
      </div>

      {/* Current Values Display */}
      <div
        style={{
          display: 'flex',
          gap: '24px',
          marginBottom: '12px',
          fontSize: '13px',
        }}
      >
        <div>
          <span style={{ color: COLORS.axis }}>Score: </span>
          <span
            style={{
              fontWeight: 600,
              color: isCurrentlyBreaching ? COLORS.breach : COLORS.normal,
            }}
          >
            {buffer[buffer.length - 1]?.composite_score.toExponential(3)}
          </span>
        </div>
        <div>
          <span style={{ color: COLORS.axis }}>Limit: </span>
          <span style={{ fontWeight: 600, color: COLORS.threshold }}>
            {currentThreshold.toExponential(3)}
          </span>
        </div>
      </div>

      {/* Chart */}
      <div style={{ width: '100%', height: '300px' }}>
        <ResponsiveContainer>
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
          >
            <defs>
              <linearGradient id="compositeGradient" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={isCurrentlyBreaching ? COLORS.breach : COLORS.normal}
                  stopOpacity={0.4}
                />
                <stop
                  offset="95%"
                  stopColor={isCurrentlyBreaching ? COLORS.breach : COLORS.normal}
                  stopOpacity={0.05}
                />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} />

            {/* X-Axis: Auto-scrolling like heart monitor */}
            <XAxis
              dataKey="relativeTime"
              domain={['dataMin', 'dataMax']}
              tickFormatter={formatXAxis}
              stroke={COLORS.axis}
              fontSize={12}
              type="number"
            />

            {/* Y-Axis: Logarithmic scale with safe minimum */}
            <YAxis
              scale="log"
              domain={yDomain}
              tickFormatter={formatYAxis}
              stroke={COLORS.axis}
              fontSize={12}
              allowDataOverflow={false}
              width={55}
            />

            <Tooltip
              formatter={(value: number, name: string) => {
                const formatted = value.toExponential(4);
                if (name === 'safeScore') return [formatted, 'Composite Score'];
                if (name === 'limit_threshold') return [formatted, 'Sigma Limit'];
                return [formatted, name];
              }}
              labelFormatter={(label: number) => {
                if (label === 0) return 'Time: Now';
                return `Time: ${label.toFixed(1)}s ago`;
              }}
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.96)',
                border: '1px solid #e5e7eb',
                borderRadius: '6px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              }}
            />

            <Legend
              formatter={(value) => {
                if (value === 'safeScore') return 'Composite Score';
                if (value === 'limit_threshold') return '3σ Limit';
                return value;
              }}
            />

            {/* Sigma Limit Reference Line */}
            <ReferenceLine
              y={currentThreshold}
              stroke={COLORS.threshold}
              strokeDasharray="3 3"
              strokeWidth={2}
              label={{
                value: `3σ Limit`,
                position: 'insideTopRight',
                fill: COLORS.threshold,
                fontSize: 11,
                fontWeight: 500,
              }}
            />

            {/* Main Signal Area */}
            <Area
              type="monotone"
              dataKey="safeScore"
              name="safeScore"
              stroke={areaStroke}
              strokeWidth={2}
              fill="url(#compositeGradient)"
              isAnimationActive={true}
              animationDuration={200}
              animationEasing="ease-out"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Pulse animation CSS */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.85; transform: scale(1.02); }
          }
        `}
      </style>
    </div>
  );
}
