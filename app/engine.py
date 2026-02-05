# engine.py
"""
Simulation Engine - Orchestrates data streaming and detection.

Handles:
- Reading data from files
- Running the detection pipeline
- Streaming results via WebSocket
- Managing simulation speed
"""

import asyncio
import os
from typing import Optional, Dict, Any
import numpy as np

from .backend import CloggingDetector
from .dataloader import DataStreamer


class SimulationEngine:
    """
    Orchestrates the simulation loop for real-time clogging detection.

    Reads data from files, runs detection algorithms, and streams
    results to connected WebSocket clients.
    """

    def __init__(self, filepath: str, speed_delay: float = 0.05):
        """
        Initialize the simulation engine.

        Args:
            filepath: Path to data file (CSV or Excel) or directory.
            speed_delay: Seconds between samples (0.05 = 20Hz simulation).
        """
        self.filepath = filepath
        self.delay = speed_delay
        self.running = False

        # Initialize detector
        self.detector = CloggingDetector()

        # Initialize data streamer
        try:
            self.streamer = DataStreamer(filepath).stream()
            self.valid = True
        except Exception as e:
            print(f"Error loading file: {e}")
            self.valid = False

    async def run_simulation(self, websocket) -> None:
        """
        Main simulation loop. Streams detection results to WebSocket.

        Args:
            websocket: WebSocket connection to stream data to.
        """
        if not self.valid:
            await websocket.send_json({"error": "Invalid file path"})
            return

        self.running = True
        print(f"Starting simulation for {self.filepath}")

        # Send initial status
        await websocket.send_json({
            "type": "status",
            "message": "Calibrating..."
        })

        calibration_buffer = []
        is_calibrating = True

        # Track available columns (sent once)
        columns_sent = False

        try:
            while self.running:
                # 1. Get next data point (now returns dict)
                try:
                    data_point = next(self.streamer)
                    t = data_point["time"]
                    flow = data_point["flow"]
                    p_in = data_point["p_in"]
                    p_out = data_point["p_out"]
                    raw_data = data_point.get("raw", {})
                    available_columns = data_point.get("columns", [])
                    dP = p_in - p_out
                except StopIteration:
                    await websocket.send_json({
                        "type": "status",
                        "message": "Simulation Complete"
                    })
                    break

                # Send column metadata once at start
                if not columns_sent and available_columns:
                    await websocket.send_json({
                        "type": "columns",
                        "columns": available_columns
                    })
                    columns_sent = True

                # 2. Calibration or Processing
                payload = {}

                if is_calibrating:
                    calibration_buffer.append(dP)
                    if len(calibration_buffer) >= 400:
                        self.detector.calibrate(calibration_buffer)
                        is_calibrating = False
                        await websocket.send_json({
                            "type": "status",
                            "message": "Calibration Done"
                        })

                    # Limited data during calibration
                    # Include null values for ensemble/models to signal "no data yet"
                    payload = {
                        "type": "data",
                        "time": t,
                        "flow": flow,
                        "pressure_drop": dP,
                        "status": "calibrating",
                        "calibration_progress": len(calibration_buffer) / 400,
                        "ensemble_probability": None,  # Signal no data, not 0%
                        "models": {},
                        "traffic_light": "gray",
                        "light_msg": f"Calibrating... {int(len(calibration_buffer) / 400 * 100)}%",
                        "raw": raw_data  # Include raw data during calibration too
                    }
                else:
                    # Run detection
                    results = self.detector.process_sample(dP, t)

                    if results:
                        # Build payload with all model results
                        payload = self._build_payload(t, flow, dP, results, raw_data)

                # 3. Send to client
                if payload:
                    await websocket.send_json(payload)

                # 4. Control simulation speed
                await asyncio.sleep(self.delay)

        except Exception as e:
            print(f"Simulation Error: {e}")
            await websocket.send_json({"error": str(e)})

    def _build_payload(self, time: float, flow: float, pressure_drop: float,
                       results: Dict[str, Any], raw_data: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Build the WebSocket payload from detection results.

        Includes all model predictions in a structured format.

        Args:
            time: Current timestamp.
            flow: Current flow rate.
            pressure_drop: Current pressure drop.
            results: Detection results from CloggingDetector.
            raw_data: Raw column data from the data file.

        Returns:
            JSON-serializable payload dict.
        """
        # Base measurements
        payload = {
            "type": "data",
            "time": time,
            "flow": flow,
            "pressure_drop": pressure_drop,

            # Legacy fields for backward compatibility
            "static_score": results.get('static', 0),
            "composite_score": results.get('composite', 0),
            "turbulence_score": results.get('turbulence', 0),
            "spectral_slope": results.get('spectral_slope', 0),
            "traffic_light": results.get('light_color', 'gray'),
            "light_msg": results.get('status_msg', ''),
            "ml_probability": results.get('ml_probability', 0),
            "limit_threshold": self.detector.fft_threshold, 
            # -----------------------------------------
            
            "turbulence_score": results.get('turbulence', 0),
            "spectral_slope": results.get('spectral_slope', 0),

            # New multi-model fields
            "models": results.get('models', {}),
            "ensemble_probability": results.get('ensemble_probability', 0),

            # Raw column data for physics visualization
            "raw": raw_data or {},
        }

        # Optionally include spectrum data (can be large, so controlled)
        # Uncomment to enable spectrum streaming:
        # if 'raw_spectrum' in results:
        #     payload['spectrum'] = {
        #         'freqs': results['raw_freqs'].tolist(),
        #         'power': results['raw_spectrum'].tolist()
        #     }

        return payload

    def stop(self) -> None:
        """Stop the simulation loop."""
        self.running = False
