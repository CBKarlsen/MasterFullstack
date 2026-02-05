# backend.py
"""
Core clogging detection logic with multi-model support.

This module provides:
- FFT-based spectral analysis
- Physics-based anomaly detection
- Integration with the model registry for ML predictions
- Sequence buffering for LSTM/sequence models
"""

import collections
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List
from sklearn.linear_model import LinearRegression

from .models import get_registry, InputType
from .models.sequence_buffer import MultiModelSequenceManager


class CloggingDetector:
    """
    Main detection engine that orchestrates multiple detection methods.

    Combines:
    - Traditional signal processing (FFT, spectral analysis)
    - Physics-based indicators (spectral slope, energy ratios)
    - Machine learning models (via model registry)
    """

    def __init__(self, fs: float = 20.0, window_sec: float = 10.0):
        """
        Initialize the clogging detector.

        Args:
            fs: Sampling frequency in Hz.
            window_sec: Analysis window size in seconds.
        """
        self.fs = fs
        self.window_size = int(window_sec * fs)  # e.g., 200 samples
        self.buffer = collections.deque(maxlen=self.window_size)
        self.composite_buffer_duration = 15.0

        # Calibration state
        self.is_calibrated = False
        self.baseline_mean = 0.0
        self.baseline_std = 0.0
        self.baseline_spectrum = None

        # Thresholds
        self.critical_threshold = 0.0
        self.fft_threshold = 0.05

        # History for plotting and prediction
        self.trend_history = []
        self.slope_buffer = []

        # Composite prediction buffers
        self.composite_buffer = []
        self.composite_times = []

        # Model registry integration
        self._registry = None

        # Sequence buffer manager for LSTM/sequence models
        self._sequence_manager = MultiModelSequenceManager()

    @property
    def registry(self):
        """Lazy load the model registry."""
        if self._registry is None:
            self._registry = get_registry()
            self._setup_sequence_buffers()
        return self._registry

    def _setup_sequence_buffers(self):
        """Initialize sequence buffers for models that need them."""
        for name, model in self.registry.get_all().items():
            if model.metadata.input_type == InputType.SEQUENCE:
                self._sequence_manager.register_model(
                    model_name=name,
                    sequence_length=model.metadata.sequence_length,
                    feature_names=model.metadata.input_features
                )

    def calibrate(self, baseline_data: List[float]) -> None:
        """
        Calibrate the detector using baseline (healthy) data.

        Args:
            baseline_data: List of pressure/signal values during normal operation.
        """
        if len(baseline_data) < 2:
            return

        self.baseline_mean = np.mean(baseline_data)
        self.baseline_std = np.std(baseline_data)

        # Static threshold
        self.critical_threshold = 10.0 * self.baseline_std

        # FFT threshold
        self.fft_threshold = 0.05

        self.is_calibrated = True
        print(f"Calibrated! Static Limit: {self.critical_threshold:.5f}, FFT Limit: {self.fft_threshold}")

        # Build baseline spectrum fingerprint
        spectra_list = []
        window_size = self.window_size
        step = int(window_size / 2)

        for i in range(0, len(baseline_data) - window_size, step):
            segment = baseline_data[i: i + window_size]
            segment = segment * np.hamming(len(segment))
            power = np.abs(np.fft.rfft(segment)) ** 2
            spectra_list.append(power)

        if spectra_list:
            self.baseline_spectrum = np.mean(spectra_list, axis=0)
            self.baseline_freqs = np.fft.rfftfreq(window_size, d=1 / self.fs)
            print("Spectral Fingerprint Calibrated!")
        else:
            self.baseline_spectrum = None

    def process_sample(self, pressure_val: float, time_sec: float) -> Optional[Dict[str, Any]]:
        """
        Process a single sample and return detection results from all methods.

        Args:
            pressure_val: Current pressure/signal value.
            time_sec: Current timestamp in seconds.

        Returns:
            Dictionary containing:
            - Basic measurements (static, composite, turbulence, spectral_slope)
            - Traffic light status
            - Model predictions from all active models
            - Raw spectrum data for visualization
        """
        self.buffer.append(pressure_val)
        results = {}

        if len(self.buffer) < self.window_size:
            return None

        # Prepare signal
        raw_signal = np.array(self.buffer)

        # =====================================================================
        # 1. STATIC METHOD (Hydraulic Head Deviation)
        # =====================================================================
        curr_mean = np.mean(raw_signal)
        static_score = 0.0
        if self.is_calibrated:
            static_score = abs(curr_mean - self.baseline_mean)
        results['static'] = static_score

        # =====================================================================
        # 2. FFT ANALYSIS
        # =====================================================================
        ham_signal = raw_signal * np.hamming(len(raw_signal))
        fft_complex = np.fft.rfft(ham_signal)
        fft_power = np.abs(fft_complex) ** 2
        freqs = np.fft.rfftfreq(len(ham_signal), 1 / self.fs)

        # Store raw spectrum for visualization
        mask_vis = freqs > 0
        results['raw_freqs'] = freqs[mask_vis]
        results['raw_spectrum'] = fft_power[mask_vis]

        # =====================================================================
        # 3. COMPOSITE METHOD (Spectral Energy Ratio)
        # =====================================================================
        split_idx = np.searchsorted(freqs, 1.0)
        low = np.sum(fft_power[:split_idx])
        high = np.sum(fft_power[split_idx:])
        composite_val = high / low if low > 1e-10 else 0.0
        results['composite'] = composite_val + 1e-6

        # =====================================================================
        # 4. TURBULENCE METHOD (Detrended FFT)
        # =====================================================================
        detrended = raw_signal - curr_mean
        detrended = detrended * np.hamming(len(detrended))
        fft_pure = np.abs(np.fft.rfft(detrended)) ** 2
        low_pure = np.sum(fft_pure[1:split_idx])
        high_pure = np.sum(fft_pure[split_idx:])
        results['turbulence'] = high_pure / low_pure if low_pure > 1e-10 else 0.0

        # =====================================================================
        # 5. ADVANCED PHYSICS & PREDICTION (Calibrated)
        # =====================================================================
        if self.is_calibrated:
            # A. Spectral Slope (Physics Indicator)
            slope_val = self.calculate_spectral_slope(freqs, fft_power)
            results['spectral_slope'] = slope_val

            # B. Composite Prediction (Traffic Light)
            light_color, status_msg = self.predict_composite_eta(time_sec, results['composite'])
            results['light_color'] = light_color
            results['status_msg'] = status_msg

            # =====================================================================
            # 6. MULTI-MODEL PREDICTIONS
            # =====================================================================
            # Prepare feature dict for models
            features = {
                'static_score': results['static'],
                'composite_score': results['composite'],
                'turbulence_score': results['turbulence'],
                'spectral_slope': slope_val,
            }

            # Update sequence buffers
            self._sequence_manager.add_features(features)

            # Run all active models
            model_results = self._run_all_models(features)
            results['models'] = model_results

            # Calculate ensemble probability (weighted average)
            results['ensemble_probability'] = self._calculate_ensemble(model_results)

            # Legacy: ML probability for backward compatibility
            if 'fft_physics' in model_results:
                results['ml_probability'] = model_results['fft_physics'].get('probability', 0.0)
            else:
                results['ml_probability'] = results['ensemble_probability']

        else:
            results['spectral_slope'] = 0.0
            results['light_color'] = "gray"
            results['status_msg'] = "Calibrating..."
            results['ml_probability'] = 0.0
            results['models'] = {}
            results['ensemble_probability'] = 0.0

        # Store history for trend analysis
        if self.is_calibrated:
            self.trend_history.append((time_sec, static_score))

        return results

    def _run_all_models(self, features: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Run prediction on all active models.

        Handles both single-input and sequence-input models.

        Args:
            features: Current feature dictionary.

        Returns:
            Dict mapping model name to prediction results.
        """
        results = {}

        for name, model in self.registry.get_active().items():
            try:
                if model.metadata.input_type == InputType.SEQUENCE:
                    # Sequence model - use buffer
                    buffer = self._sequence_manager.get_buffer(name)
                    if buffer and buffer.is_ready():
                        seq_features = {'sequence': buffer.get_sequence()}
                        result = model.predict(seq_features)
                        results[name] = result.to_dict()
                        results[name]['buffer_status'] = 'ready'
                    else:
                        # Not enough data yet
                        fill = buffer.fill_ratio() if buffer else 0.0
                        results[name] = {
                            'probability': 0.0,
                            'confidence': 0.0,
                            'buffer_status': f'filling ({fill:.0%})'
                        }
                else:
                    # Single-input model
                    result = model.predict(features)
                    results[name] = result.to_dict()

            except Exception as e:
                results[name] = {
                    'probability': 0.0,
                    'confidence': 0.0,
                    'error': str(e)
                }

        return results

    def _calculate_ensemble(self, model_results: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate weighted ensemble probability from all model predictions.

        Uses confidence-weighted average.

        Args:
            model_results: Results from all models.

        Returns:
            Ensemble probability (0.0 - 1.0).
        """
        probabilities = []
        weights = []

        for name, result in model_results.items():
            prob = result.get('probability', 0.0)
            conf = result.get('confidence', 1.0)

            if 'error' not in result and prob > 0:
                probabilities.append(prob)
                weights.append(conf)

        if not probabilities:
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            return sum(p * w for p, w in zip(probabilities, weights)) / total_weight
        else:
            return np.mean(probabilities)

    def calculate_spectral_slope(self, freqs: np.ndarray, power: np.ndarray) -> float:
        """
        Calculate the spectral decay rate (log-log slope).

        Physics basis:
        - Healthy Flow: Steep slope (~ -2.5 to -3.0) following Kolmogorov cascade
        - Clogged Flow: Flat slope (~ -1.0 to -1.5) due to white noise/cavitation

        Args:
            freqs: Frequency array from FFT.
            power: Power spectrum array.

        Returns:
            Spectral slope value.
        """
        # Filter for relevant frequency band (1-50 Hz)
        mask = (freqs > 1.0) & (freqs < 50.0)

        if np.sum(mask) < 5:
            return 0.0

        try:
            x = np.log10(freqs[mask])
            y = np.log10(power[mask])
            slope, _ = np.polyfit(x, y, 1)
            return slope
        except Exception:
            return 0.0

    def predict_robust(self) -> Optional[tuple]:
        """
        Robust linear prediction of time to critical threshold.

        Returns:
            Tuple of (time_remaining_hours, slope, intercept, r2, is_reliable)
            or None if not enough data.
        """
        if len(self.trend_history) < 500:
            return None

        data = list(self.trend_history)
        times = np.array([x[0] for x in data])
        scores = np.array([x[1] for x in data])

        # Smoothing
        window_len = 50
        if len(scores) > window_len:
            scores_smooth = np.convolve(scores, np.ones(window_len) / window_len, mode='valid')
            times_smooth = times[window_len - 1:]
        else:
            scores_smooth = scores
            times_smooth = times

        # Linear regression
        X = times_smooth[-1000:].reshape(-1, 1)
        y = scores_smooth[-1000:]

        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        r2 = model.score(X, y)

        # Median filter for stability
        self.slope_buffer.append(slope)
        stable_slope = np.median(self.slope_buffer)

        if stable_slope <= 0:
            return None

        current_val = scores_smooth[-1]
        time_rem_sec = (self.critical_threshold - current_val) / stable_slope
        time_rem_hours = time_rem_sec / 3600.0

        stable_intercept = current_val - (stable_slope * times_smooth[-1])
        is_reliable = (r2 >= 0.6)

        return (time_rem_hours, stable_slope, stable_intercept, r2, is_reliable)

    def predict_composite_eta(self, current_time: float, current_score: float) -> tuple:
        """
        Calculate time to critical using log-linear regression.

        Args:
            current_time: Current timestamp.
            current_score: Current composite score.

        Returns:
            Tuple of (traffic_light_color, status_message).
        """
        # Update buffer
        self.composite_buffer.append(current_score)
        self.composite_times.append(current_time)

        # Keep buffer manageable (~60 seconds)
        if len(self.composite_buffer) > 300:
            self.composite_buffer.pop(0)
            self.composite_times.pop(0)

        # Need enough data
        if len(self.composite_buffer) < 50:
            return "gray", "Initializing..."

        # Define thresholds
        CRITICAL_LIMIT = self.fft_threshold
        WARNING_LEVEL = CRITICAL_LIMIT * 0.2

        # Green zone check
        if current_score < WARNING_LEVEL:
            return "green", "System Stable"

        # Log-linear regression
        try:
            subset_scores = np.array(self.composite_buffer)[-150:]
            subset_times = np.array(self.composite_times)[-150:]

            # Avoid log(0)
            subset_scores = np.maximum(subset_scores, 1e-9)
            log_scores = np.log(subset_scores)

            slope, intercept = np.polyfit(subset_times, log_scores, 1)

            if slope <= 0:
                return "green", "Stable (No Growth)"

            # Time to impact
            target_log = np.log(CRITICAL_LIMIT)
            current_log = log_scores[-1]
            seconds_left = (target_log - current_log) / slope

            # Decide output
            if seconds_left > 1200:  # >20 min
                return "green", "Slight Trend (>20m)"
            elif seconds_left > 300:  # 5-20 min
                return "yellow", f"Warning: ~{int(seconds_left / 60)} min left"
            elif seconds_left > 0:
                return "red", f"CRITICAL: < {int(seconds_left)}s"
            else:
                return "red", "FAILURE IMMINENT"

        except Exception:
            return "gray", "Calc Error"
