import type { TrafficLight as TLType } from '../types';

interface TrafficLightProps {
  color: TLType;
  message?: string;
}

export function TrafficLight({ color, message }: TrafficLightProps) {
  const colors: Record<TLType, { bg: string; text: string }> = {
    green: { bg: '#22c55e', text: 'Healthy' },
    yellow: { bg: '#eab308', text: 'Warning' },
    red: { bg: '#ef4444', text: 'Critical' },
    gray: { bg: '#6b7280', text: 'Unknown' },
  };

  const { bg, text } = colors[color] || colors.gray;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
      <div
        style={{
          width: '48px',
          height: '48px',
          borderRadius: '50%',
          backgroundColor: bg,
          boxShadow: `0 0 20px ${bg}`,
          transition: 'all 0.3s ease',
        }}
      />
      <div>
        <div style={{ fontWeight: 'bold', fontSize: '18px' }}>{text}</div>
        {message && (
          <div style={{ fontSize: '14px', color: '#666' }}>{message}</div>
        )}
      </div>
    </div>
  );
}
