# engine.py
"""
Simulation engine that orchestrates the detection loop.

CHANGES:
- Passes sigma to CloggingDetector
- Forwards fft_threshold and static_threshold in every payload
- Handles incoming sigma change messages from WebSocket client
- Forwards current_sigma so frontend stays in sync
"""

import asyncio
import os
from typing import Optional, Dict, Any
import numpy as np
import traceback

from .backend import CloggingDetector
from .dataloader import DataStreamer


class SimulationEngine:
    """
    Orchestrates the simulation loop for real-time clogging detection.
    """

    def __init__(self, filepath: str, speed_delay: float = 0.1, sigma: float = 3.0):
        self.filepath = filepath
        self.delay = speed_delay
        self.running = False
        self.detector = CloggingDetector(sigma=sigma)
        self._frame_count = 0

        try:
            self.streamer = DataStreamer(filepath).stream()
            self.valid = True
        except Exception as e:
            print(f"Error initializing data streamer: {e}")
            self.valid = False

    def update_sigma(self, new_sigma: float) -> Dict[str, float]:
        """
        Update sigma on the detector and return new thresholds.

        Call this when the frontend sends a sigma change request.
        """
        return self.detector.set_sigma(new_sigma)

    async def run_simulation(self, websocket) -> None:
        """
        Main simulation loop. Streams detection results to WebSocket.
        """
        if not self.valid:
            await websocket.send_json({"error": "Invalid file path"})
            return

        self.running = True
        print(f"Starting simulation for {self.filepath}")

        await websocket.send_json({
            "type": "status",
            "message": "Calibrating..."
        })

        calibration_buffer = []
        is_calibrating = True
        columns_sent = False

        try:
            while self.running:
                # --- Data Fetching ---
                try:
                    data_point = next(self.streamer)
                except StopIteration:
                    print("Stream finished (StopIteration)")
                    await websocket.send_json({
                        "type": "status",
                        "message": "Simulation Complete"
                    })
                    break
                except Exception as e:
                    print(f"Skipping bad data point: {e}")
                    await asyncio.sleep(0)
                    continue

                # Extract data
                t = data_point.get("time", 0.0)
                flow = data_point.get("flow", 0.0)
                p_in = data_point.get("p_in", 0.0)
                p_out = data_point.get("p_out", 0.0)
                raw_data = data_point.get("raw", {})
                available_columns = data_point.get("columns", [])

                dP = p_in - p_out

                # Send column metadata once
                if not columns_sent and available_columns:
                    await websocket.send_json({
                        "type": "columns",
                        "columns": available_columns
                    })
                    columns_sent = True

                payload = {}

                # Calibration or Processing
                if is_calibrating:
                    calibration_buffer.append(dP)
                    if len(calibration_buffer) >= 400:
                        try:
                            self.detector.calibrate(calibration_buffer)
                            is_calibrating = False
                            await websocket.send_json({
                                "type": "status",
                                "message": "Calibration Done"
                            })
                            # Send initial thresholds after calibration
                            await websocket.send_json({
                                "type": "thresholds",
                                "sigma": self.detector.sigma,
                                "fft_threshold": self.detector.fft_threshold,
                                "static_threshold": self.detector.critical_threshold,
                                "composite_baseline_mean": self.detector.baseline_composite_mean,
                                "composite_baseline_std": self.detector.baseline_composite_std,
                            })
                        except Exception as e:
                            print(f"Calibration failed: {e}")
                            calibration_buffer = []

                    payload = {
                        "type": "data",
                        "time": t,
                        "flow": flow,
                        "pressure_drop": dP,
                        "status": "calibrating",
                        "limit_threshold": self.detector.fft_threshold,
                        "static_threshold": self.detector.critical_threshold,
                        "current_sigma": self.detector.sigma,
                        "traffic_light": "gray",
                        "raw": raw_data
                    }
                else:
                    # Run detection
                    try:
                        results = self.detector.process_sample(dP, t)
                        if results:
                            payload = self._build_payload(t, flow, dP, results, raw_data)
                    except Exception as e:
                        print(f"Detection Error at t={t}: {e}")
                        payload = {
                            "type": "data",
                            "time": t,
                            "flow": flow,
                            "pressure_drop": dP,
                            "raw": raw_data,
                            "error": "Calculation Failed"
                        }

                # Send to client
                if payload:
                    try:
                        self._frame_count += 1
                        if self._frame_count % 10 == 0:  # Send every 10th frame = 2Hz instead of 20Hz
                            await websocket.send_json(payload)
                    except Exception as e:
                        print(f"WebSocket Send Error: {e}")
                        break

                await asyncio.sleep(self.delay)

        except Exception as e:
            print(f"CRITICAL SIMULATION CRASH: {e}")
            traceback.print_exc()
            try:
                await websocket.send_json({"error": f"Server Crash: {str(e)}"})
            except:
                pass

    def _build_payload(self, time: float, flow: float, pressure_drop: float,
                       results: Dict[str, Any], raw_data: Dict[str, float] = None) -> Dict[str, Any]:

        payload = {
            "type": "data",
            "time": time,
            "flow": flow,
            "pressure_drop": pressure_drop,

            # Scores
            "composite_score": results.get('composite', 0),
            "static_score": results.get('static', 0),
            "spectral_slope": results.get('spectral_slope', 0),
            "turbulence_score": results.get('turbulence', 0),
            "wavelet_score": results.get('wavelet', 0),

            # Thresholds (sent every frame so frontend stays in sync)
            "limit_threshold": results.get('fft_threshold', self.detector.fft_threshold),
            "static_threshold": results.get('static_threshold', self.detector.critical_threshold),
            "wavelet_threshold": results.get('wavelet_threshold', self.detector.wavelet_threshold),
            "current_sigma": results.get('current_sigma', self.detector.sigma),

            # Traffic light
            "traffic_light": results.get('light_color', 'gray'),
            "light_msg": results.get('status_msg', ''),

            # Models
            "models": results.get('models', {}),
            "ensemble_probability": results.get('ensemble_probability', 0),

            # Raw sensor data
            "raw": raw_data or {},
        }
        return payload

    def stop(self) -> None:
        self.running = False