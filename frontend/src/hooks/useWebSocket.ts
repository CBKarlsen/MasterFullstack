/**
 * useWebSocket — manages the live connection to the backend simulation stream.
 *
 * Opens a WebSocket to the simulation endpoint, sends the start/stop control
 * messages, and surfaces connection status and the most recent frame to the
 * UI. The backend pushes detection frames at roughly 20 Hz (and faster under
 * the speed multiplier), so each incoming message is pushed into a buffer and
 * the buffer is flushed once per `requestAnimationFrame` (~60 Hz cap). This
 * decouples the wire rate from React renders: instead of one store write per
 * message, consumers receive a single batched callback (`onBatch`) per frame.
 *
 * Inputs: the WebSocket `url` plus optional `onMessage` / `onBatch` handlers
 * (kept in refs so the long-lived socket callbacks never read stale closures).
 * Outputs: `isConnected`, `lastMessage` (latest frame only), `error`, and the
 * `startSimulation` / `stopSimulation` controls. The buffer is capped at
 * MAX_BUFFERED_MESSAGES so a slow renderer at high speed cannot exhaust memory.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import type { SimulationConfig, SimulationData } from "../types";

// Hard cap on the unflushed buffer; older messages are dropped past this.
const MAX_BUFFERED_MESSAGES = 500;

interface UseWebSocketOptions {
	onMessage?: (data: SimulationData) => void;
	onBatch?: (batch: SimulationData[]) => void;
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
	options: UseWebSocketOptions = {},
): UseWebSocketReturn {
	const { onMessage, onBatch } = options;

	const wsRef = useRef<WebSocket | null>(null);
	// Keep stable refs so the WebSocket callbacks never go stale
	const onMessageRef = useRef(onMessage);
	const onBatchRef = useRef(onBatch);
	useEffect(() => {
		onMessageRef.current = onMessage;
	}, [onMessage]);
	useEffect(() => {
		onBatchRef.current = onBatch;
	}, [onBatch]);

	// Message buffer: accumulate incoming messages and flush once per animation frame
	const bufferRef = useRef<SimulationData[]>([]);
	const rafRef = useRef<number | null>(null);

	const [isConnected, setIsConnected] = useState(false);
	const [lastMessage, setLastMessage] = useState<SimulationData | null>(null);
	const [error, setError] = useState<string | null>(null);

	// Flush buffered messages at display refresh rate instead of 20Hz wire rate
	const scheduleFlush = useCallback(() => {
		if (rafRef.current !== null) return; // already scheduled
		rafRef.current = requestAnimationFrame(() => {
			rafRef.current = null;
			const buf = bufferRef.current;
			if (buf.length === 0) return;

			bufferRef.current = [];

			// Prefer a single batched callback to avoid per-message store writes
			if (onBatchRef.current) {
				onBatchRef.current(buf);
			} else if (onMessageRef.current) {
				for (const msg of buf) {
					onMessageRef.current(msg);
				}
			}
			// Only the newest frame goes into React state; the full batch was
			// already handed to the consumer above, so re-rendering on every
			// buffered message would be wasted work.
			setLastMessage(buf[buf.length - 1]);
		});
	}, []);

	const startSimulation = useCallback(
		(file: string, speed: number = 1.0) => {
			// Close existing connection if any
			if (wsRef.current) {
				wsRef.current.close();
			}

			try {
				// Build WebSocket URL
				const wsUrl = url.startsWith("/")
					? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}${url}`
					: url;

				wsRef.current = new WebSocket(wsUrl);
				setError(null);

				wsRef.current.onopen = () => {
					console.log("WebSocket connected, starting simulation...");
					setIsConnected(true);
					setError(null);

					// Send start configuration
					const config: SimulationConfig = {
						action: "start",
						file,
						speed,
					};
					wsRef.current?.send(JSON.stringify(config));
				};

				wsRef.current.onmessage = (event) => {
					try {
						const data: SimulationData = JSON.parse(event.data);

						// Check for error response
						if ("error" in data) {
							setError(data.error as string);
							return;
						}

						// Buffer the message and schedule a batched flush.
						// Cap the buffer so a slow renderer can't OOM the tab at high speeds.
						bufferRef.current.push(data);
						if (bufferRef.current.length > MAX_BUFFERED_MESSAGES) {
							// Drop the oldest overflow so only the most recent
							// MAX_BUFFERED_MESSAGES frames survive until the next flush.
							bufferRef.current.splice(
								0,
								bufferRef.current.length - MAX_BUFFERED_MESSAGES,
							);
						}
						scheduleFlush();
					} catch (e) {
						console.error("Failed to parse WebSocket message:", e);
					}
				};

				wsRef.current.onerror = (event) => {
					console.error("WebSocket error:", event);
					setError("Connection error");
				};

				wsRef.current.onclose = () => {
					console.log("WebSocket disconnected");
					setIsConnected(false);
				};
			} catch (e) {
				setError(`Failed to connect: ${e}`);
			}
		},
		[url, scheduleFlush],
	);

	const stopSimulation = useCallback(() => {
		if (wsRef.current) {
			// Send stop command
			try {
				wsRef.current.send(JSON.stringify({ action: "stop" }));
			} catch {
				// Ignore send errors on close
			}
			wsRef.current.close();
			wsRef.current = null;
		}
		setIsConnected(false);
	}, []);

	// Clean up WebSocket and pending animation frame on unmount
	useEffect(() => {
		return () => {
			if (rafRef.current !== null) {
				cancelAnimationFrame(rafRef.current);
			}
			if (wsRef.current) {
				wsRef.current.close();
				wsRef.current = null;
			}
		};
	}, []);

	return {
		isConnected,
		lastMessage,
		startSimulation,
		stopSimulation,
		error,
	};
}
