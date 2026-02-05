import { useRef, useState, useCallback } from 'react';
import type { SimulationData, SimulationConfig } from '../types';

interface UseWebSocketOptions {
  onMessage?: (data: SimulationData) => void;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  lastMessage: SimulationData | null;
  startSimulation: (file: string, speed?: number) => void;
  stopSimulation: () => void;
  error: string | null;
}

export function useWebSocket(
  url: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const { onMessage } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<SimulationData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const startSimulation = useCallback((file: string, speed: number = 1.0) => {
    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      // Build WebSocket URL
      const wsUrl = url.startsWith('/')
        ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}${url}`
        : url;

      wsRef.current = new WebSocket(wsUrl);
      setError(null);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected, starting simulation...');
        setIsConnected(true);
        setError(null);

        // Send start configuration
        const config: SimulationConfig = {
          action: 'start',
          file,
          speed,
        };
        wsRef.current?.send(JSON.stringify(config));
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data: SimulationData = JSON.parse(event.data);

          // Check for error response
          if ('error' in data) {
            setError(data.error as string);
            return;
          }

          setLastMessage(data);
          onMessage?.(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      wsRef.current.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('Connection error');
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
      };
    } catch (e) {
      setError(`Failed to connect: ${e}`);
    }
  }, [url, onMessage]);

  const stopSimulation = useCallback(() => {
    if (wsRef.current) {
      // Send stop command
      try {
        wsRef.current.send(JSON.stringify({ action: 'stop' }));
      } catch {
        // Ignore send errors on close
      }
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  return {
    isConnected,
    lastMessage,
    startSimulation,
    stopSimulation,
    error,
  };
}
