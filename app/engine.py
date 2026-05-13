# engine.py
"""Simulation engine that orchestrates the detection loop."""

import asyncio
import traceback
from typing import Optional, Dict, Any, Tuple

import numpy as np

from .backend import CloggingDetector
from .dataloader import DataStreamer

TARGET_WIRE_HZ = 2.0  # Approximate WebSocket frame rate to the client
DEFAULT_SAMPLING_HZ = 20.0  # Fallback when fs cannot be detected from timestamps
CALIBRATION_FS_LOCK_AFTER = 10  # Detect fs after this many samples to size calibration
MIN_VALID_DELTA_SEC = 0.0  # Exclude zero-length deltas (duplicate rows)
MAX_VALID_DELTA_SEC = 60.0  # Exclude huge gaps (file breaks, missing data)


def detect_sampling_rate(times: list) -> float:
    """Infer sampling frequency from median of timestamp deltas.

    Shared by real-time (engine) and batch paths so both observe the same fs
    for a given file. Uses the first 100 deltas, ignoring zero-length and
    gap-sized deltas.
    """
    if len(times) < 2:
        return DEFAULT_SAMPLING_HZ
    deltas = np.diff(times[:100])
    valid = deltas[(deltas > MIN_VALID_DELTA_SEC) & (deltas < MAX_VALID_DELTA_SEC)]
    if len(valid) == 0:
        return DEFAULT_SAMPLING_HZ
    return round(1.0 / float(np.median(valid)), 2)


class SimulationEngine:
    """Orchestrates the simulation loop for real-time clogging detection."""

    def __init__(
        self, filepath: str, speed_multiplier: float = 1.0, sigma: float = 3.0
    ):
        self.filepath = filepath
        self.speed_multiplier = max(0.1, min(speed_multiplier, 100.0))
        # Provisional delay until calibration determines the true fs. Assumes 20 Hz.
        self.delay = (1.0 / DEFAULT_SAMPLING_HZ) / self.speed_multiplier
        self.frame_skip_rate = max(1, int(DEFAULT_SAMPLING_HZ / TARGET_WIRE_HZ))
        self.running = False
        self._sigma = sigma
        self.detector = CloggingDetector(sigma=sigma)
        self._frame_count = 0
        self._required_calibration_samples = (
            CloggingDetector.required_calibration_samples(DEFAULT_SAMPLING_HZ)
        )

        try:
            self.streamer = DataStreamer(filepath).stream()
            self.valid = True
        except Exception as e:
            print(f"Error initializing data streamer: {e}")
            self.valid = False

    def update_sigma(self, new_sigma: float) -> Dict[str, float]:
        """Update sigma on the detector and return new thresholds."""
        return self.detector.set_sigma(new_sigma)

    async def run_simulation(self, websocket) -> None:
        """Main simulation loop. Streams detection results to WebSocket."""
        if not self.valid:
            await websocket.send_json({"error": "Invalid file path"})
            return

        self.running = True
        calibration_buffer: list = []
        calibration_times: list = []
        is_calibrating = True
        columns_sent = False
        delay_locked = False

        await websocket.send_json({"type": "status", "message": "Calibrating..."})

        try:
            while self.running:
                data_point, is_done = self._next_data_point()
                if is_done:
                    await websocket.send_json(
                        {"type": "status", "message": "Simulation Complete"}
                    )
                    break
                if data_point is None:
                    await asyncio.sleep(0)
                    continue

                t, flow, dP, raw_data, available_columns = self._extract_fields(
                    data_point
                )

                if not columns_sent and available_columns:
                    await websocket.send_json(
                        {"type": "columns", "columns": available_columns}
                    )
                    columns_sent = True

                if is_calibrating:
                    calibration_buffer.append(dP)
                    calibration_times.append(t)
                    # Lock in real-time pacing as soon as we have enough timestamps
                    # to estimate fs — well before calibration completes.
                    if (
                        not delay_locked
                        and len(calibration_times) >= CALIBRATION_FS_LOCK_AFTER
                    ):
                        detected_fs = detect_sampling_rate(calibration_times)
                        self.delay = (1.0 / detected_fs) / self.speed_multiplier
                        # Scale frame-skip by both fs and speed so the wire rate
                        # stays near TARGET_WIRE_HZ at every playback speed.
                        # Without the speed factor, 20x bursts ~40 Hz to the
                        # browser and OOM-crashes the tab.
                        effective_hz = detected_fs * self.speed_multiplier
                        self.frame_skip_rate = max(1, int(effective_hz / TARGET_WIRE_HZ))
                        # Size calibration buffer to fit ≥21 spectral windows at this fs.
                        self._required_calibration_samples = (
                            CloggingDetector.required_calibration_samples(detected_fs)
                        )
                        delay_locked = True
                        print(
                            f"Inter-sample delay locked: fs={detected_fs} Hz, "
                            f"delay={self.delay:.4f}s ({self.speed_multiplier}x), "
                            f"frame_skip={self.frame_skip_rate}, "
                            f"calibration_samples={self._required_calibration_samples}"
                        )
                    is_calibrating = await self._maybe_finish_calibration(
                        websocket, calibration_buffer, calibration_times
                    )
                    payload = self._build_calibrating_payload(t, flow, dP, raw_data)
                else:
                    payload = self._build_detection_payload(t, flow, dP, raw_data)

                await self._maybe_send(websocket, payload)
                await asyncio.sleep(self.delay)

        except Exception as e:
            print(f"CRITICAL SIMULATION CRASH: {e}")
            traceback.print_exc()
            try:
                await websocket.send_json({"error": f"Server Crash: {str(e)}"})
            except Exception:
                pass

    def _next_data_point(self) -> Tuple[Optional[Dict], bool]:
        """Advance the stream. Returns (data_point | None, is_done)."""
        try:
            return next(self.streamer), False
        except StopIteration:
            print("Stream finished (StopIteration)")
            return None, True
        except Exception as e:
            print(f"Skipping bad data point: {e}")
            return None, False

    def _extract_fields(
        self, data_point: Dict
    ) -> Tuple[float, float, float, Dict, list]:
        t = data_point.get("time", 0.0)
        flow = data_point.get("flow", 0.0)
        p_in = data_point.get("p_in", 0.0)
        p_out = data_point.get("p_out", 0.0)
        raw_data = data_point.get("raw", {})
        available_columns = data_point.get("columns", [])
        return t, flow, p_in - p_out, raw_data, available_columns

    async def _maybe_finish_calibration(
        self, websocket, buffer: list, times: list
    ) -> bool:
        """Run calibration when enough samples are collected. Returns is_still_calibrating."""
        if len(buffer) < self._required_calibration_samples:
            return True
        try:
            detected_fs = detect_sampling_rate(times)
            self.detector = CloggingDetector(fs=detected_fs, sigma=self._sigma)
            # Re-pace inter-sample delay against the actual sampling rate.
            self.delay = (1.0 / detected_fs) / self.speed_multiplier
            print(
                f"Detector initialized with fs={detected_fs} Hz, "
                f"delay={self.delay:.4f}s ({self.speed_multiplier}x)"
            )

            self.detector.calibrate(buffer)
            await websocket.send_json({"type": "status", "message": "Calibration Done"})
            await websocket.send_json(
                {
                    "type": "thresholds",
                    "sigma": self.detector.sigma,
                    "fft_threshold": self.detector.fft_threshold,
                    "static_threshold": self.detector.critical_threshold,
                    "composite_baseline_mean": self.detector.baseline_composite_mean,
                    "composite_baseline_std": self.detector.baseline_composite_std,
                    "sampling_hz": detected_fs,
                }
            )
        except Exception as e:
            print(f"Calibration failed: {e}")
            buffer.clear()
            times.clear()
            return True
        return False

    def _build_calibrating_payload(
        self, t: float, flow: float, dP: float, raw_data: Dict
    ) -> Dict:
        return {
            "type": "data",
            "time": t,
            "flow": flow,
            "pressure_drop": dP,
            "status": "calibrating",
            "limit_threshold": self.detector.fft_threshold,
            "static_threshold": self.detector.critical_threshold,
            "current_sigma": self.detector.sigma,
            "traffic_light": "gray",
            "raw": raw_data,
        }

    def _build_detection_payload(
        self, t: float, flow: float, dP: float, raw_data: Dict
    ) -> Dict:
        try:
            results = self.detector.process_sample(dP, t)
            if results:
                return self._build_payload(t, flow, dP, results, raw_data)
            # Buffer is still warming up after calibration (window_size samples needed).
            buffer_pct = int(
                100 * len(self.detector.buffer) / self.detector.window_size
            )
            return {
                "type": "data",
                "time": t,
                "flow": flow,
                "pressure_drop": dP,
                "status": "buffering",
                "buffer_pct": buffer_pct,
                "limit_threshold": self.detector.fft_threshold,
                "static_threshold": self.detector.critical_threshold,
                "current_sigma": self.detector.sigma,
                "traffic_light": "gray",
                "raw": raw_data,
            }
        except Exception as e:
            print(f"Detection Error at t={t}: {e}")
            return {
                "type": "data",
                "time": t,
                "flow": flow,
                "pressure_drop": dP,
                "raw": raw_data,
                "error": "Calculation Failed",
            }

    async def _maybe_send(self, websocket, payload: Dict) -> None:
        """Send payload on every Nth frame to reduce wire rate."""
        if not payload:
            return
        self._frame_count += 1
        if self._frame_count % self.frame_skip_rate != 0:
            return
        try:
            await websocket.send_json(payload)
        except Exception as e:
            print(f"WebSocket Send Error: {e}")
            self.running = False

    def _build_payload(
        self,
        time: float,
        flow: float,
        pressure_drop: float,
        results: Dict[str, Any],
        raw_data: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        return {
            "type": "data",
            "time": time,
            "flow": flow,
            "pressure_drop": pressure_drop,
            "composite_score": results.get("composite", 0),
            "static_score": results.get("static", 0),
            "spectral_slope": results.get("spectral_slope", 0),
            "turbulence_score": results.get("turbulence", 0),
            "limit_threshold": results.get(
                "fft_threshold", self.detector.fft_threshold
            ),
            "static_threshold": results.get(
                "static_threshold", self.detector.critical_threshold
            ),
            "current_sigma": results.get("current_sigma", self.detector.sigma),
            "traffic_light": results.get("light_color", "gray"),
            "light_msg": results.get("status_msg", ""),
            "models": results.get("models", {}),
            "ensemble_probability": results.get("ensemble_probability", 0),
            "raw": raw_data or {},
        }

    def stop(self) -> None:
        self.running = False
