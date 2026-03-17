# backend.py
"""
Core clogging detection logic with multi-model support.

This module provides:
- FFT-based spectral analysis
- Physics-based anomaly detection
- Integration with the model registry for ML predictions
- Sequence buffering for LSTM/sequence models

CHANGES:
- fft_threshold is now computed from calibration baseline (sigma-based)
- sigma_multiplier is configurable at init and via set_sigma()
- static threshold also uses configurable sigma
- Thresholds are recalculated when sigma changes
"""

import collections
import math
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

    def __init__(self, fs: float = 20.0, window_sec: float = 10.0,
                 sigma: float = 3.0, enable_models: bool = True):
        """
        Initialize the clogging detector.

        Args:
            fs: Sampling frequency in Hz.
            window_sec: Analysis window size in seconds.
            sigma: Number of standard deviations for threshold calculation.
        """
        self.fs = fs
        self.window_size = int(window_sec * fs)  # e.g., 200 samples
        self.buffer = collections.deque(maxlen=self.window_size)
        self.composite_buffer_duration = 15.0

        # Sigma multiplier (configurable)
        self.sigma = sigma

        # Calibration state
        self.is_calibrated = False
        self.baseline_mean = 0.0
        self.baseline_std = 0.0
        self.baseline_spectrum = None

        # Baseline composite stats (for sigma-based FFT threshold)
        self.baseline_composite_mean = 0.0
        self.baseline_composite_std = 0.0
        self.baseline_static_mean = 0.0
        self.baseline_static_std = 0.0

        # Thresholds (computed during calibration)
        self.critical_threshold = 0.0
        self.fft_threshold = 0.05  # fallback default

        # History for plotting and prediction
        self.trend_history = []
        self.slope_buffer = []

        # Composite prediction buffers
        self.composite_buffer = []
        self.composite_times = []

        # Model registry integration
        self._registry = None
        self._enable_models = enable_models

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

    def set_sigma(self, new_sigma: float) -> Dict[str, float]:
        """
        Update sigma multiplier and recalculate thresholds.

        Args:
            new_sigma: New sigma value (e.g. 2.0, 3.0, 5.0).

        Returns:
            Dict with updated threshold values.
        """
        self.sigma = new_sigma
        if self.is_calibrated:
            self._recalculate_thresholds()
        return {
            'sigma': self.sigma,
            'fft_threshold': self.fft_threshold,
            'critical_threshold': self.critical_threshold,
        }

    def _recalculate_thresholds(self):
        """Recalculate all thresholds from stored baseline stats + current sigma."""
        # Static threshold: baseline_mean + sigma * baseline_std of dP
        self.critical_threshold = self.sigma * self.baseline_std

        # FFT/Composite threshold: baseline composite mean + sigma * composite std
        if self.baseline_composite_std > 0:
            self.fft_threshold = (self.baseline_composite_mean
                                  + self.sigma * self.baseline_composite_std)
        elif self.baseline_composite_mean > 0:
            # No variance in baseline — use multiplier of mean
            self.fft_threshold = self.baseline_composite_mean * (1.0 + self.sigma)
        else:
            # No composite data at all — use relative fallback
            self.fft_threshold = 0.05

        print(f"Thresholds recalculated (σ={self.sigma:.1f}): "
              f"Static={self.critical_threshold:.5f}, "
              f"Composite={self.fft_threshold:.5f}")

    def calibrate(self, baseline_data: List[float]) -> None:
        """
        Calibrate the detector using baseline (healthy) data.

        Computes baseline statistics for both raw signal and derived
        composite scores, then sets thresholds based on sigma.

        Args:
            baseline_data: List of pressure/signal values during normal operation.
        """
        if len(baseline_data) < 2:
            return

        self.baseline_mean = np.mean(baseline_data)
        self.baseline_std = np.std(baseline_data)

        # --- Compute baseline composite scores from calibration windows ---
        # This is the key fix: we run the same FFT composite calculation on
        # the calibration data to establish what "normal" composite values
        # look like, then set the threshold at mean + sigma * std.
        baseline_composites = []
        window_size = self.window_size
        step = max(1, window_size // 20)  # ~5% step → ~20x more windows than 50% overlap

        spectra_list = []

        for i in range(0, len(baseline_data) - window_size + 1, step):
            segment = np.array(baseline_data[i: i + window_size])

            # FFT composite (same logic as process_sample)
            ham_signal = segment * np.hamming(len(segment))
            fft_power = np.abs(np.fft.rfft(ham_signal)) ** 2
            freqs = np.fft.rfftfreq(len(ham_signal), 1 / self.fs)

            nyquist = self.fs / 2.0
            split_freq = min(1.0, nyquist / 4.0)  # Split at 25% of Nyquist
            split_idx = max(2, np.searchsorted(freqs, split_freq))  # At least 2 bins in low band
            total = np.sum(fft_power)
            composite_val = np.sum(fft_power[split_idx:]) / total if total > 1e-10 else 0.0
            baseline_composites.append(composite_val)

            # Also build spectral fingerprint
            spectra_list.append(fft_power)

        # Store baseline composite statistics
        if baseline_composites:
            self.baseline_composite_mean = float(np.mean(baseline_composites))
            self.baseline_composite_std = float(np.std(baseline_composites))
        else:
            self.baseline_composite_mean = 0.0
            self.baseline_composite_std = 0.0

        # Store baseline static statistics
        self.baseline_static_mean = 0.0  # deviation from mean is ~0 at baseline
        self.baseline_static_std = self.baseline_std

        self.is_calibrated = True

        # Calculate thresholds using current sigma
        self._recalculate_thresholds()

        print(f"Calibrated! Baseline composite: "
              f"mean={self.baseline_composite_mean:.6f}, "
              f"std={self.baseline_composite_std:.6f}")
        print(f"  → fft_threshold (σ={self.sigma}): {self.fft_threshold:.6f}")

        # Build baseline spectrum fingerprint
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
            Dictionary containing all detection results including thresholds.
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
        results['raw_freqs'] = freqs[mask_vis].tolist()
        results['raw_spectrum'] = fft_power[mask_vis].tolist()

        # =====================================================================
        # 3. COMPOSITE METHOD (Spectral Energy Ratio)
        # =====================================================================
        nyquist = self.fs / 2.0
        split_freq = min(1.0, nyquist / 4.0)
        split_idx = max(2, np.searchsorted(freqs, split_freq))
        total_power = np.sum(fft_power)
        composite_val = np.sum(fft_power[split_idx:]) / total_power if total_power > 1e-10 else 0.0
        results['composite'] = composite_val

        # =====================================================================
        # 4. TURBULENCE METHOD (Detrended FFT)
        # =====================================================================
        detrended = raw_signal - curr_mean
        detrended = detrended * np.hamming(len(detrended))
        fft_pure = np.abs(np.fft.rfft(detrended)) ** 2
        total_pure = np.sum(fft_pure[1:])
        results['turbulence'] = np.sum(fft_pure[split_idx:]) / total_pure if total_pure > 1e-10 else 0.0

        # =====================================================================
        # 5. ADVANCED PHYSICS & PREDICTION (Calibrated)
        # =====================================================================
        if self.is_calibrated:
            # A. Spectral Slope
            slope_val = self.calculate_spectral_slope(freqs, fft_power)
            results['spectral_slope'] = slope_val

            # B. Composite Prediction (Traffic Light)
            light_color, status_msg = self.predict_composite_eta(time_sec, results['composite'])
            results['light_color'] = light_color
            results['status_msg'] = status_msg

            # C. Current thresholds (so frontend always has latest)
            results['fft_threshold'] = self.fft_threshold
            results['static_threshold'] = self.critical_threshold
            results['current_sigma'] = self.sigma

            # =====================================================================
            # 6. MULTI-MODEL PREDICTIONS
            # =====================================================================
            features = {
                'static_score': results['static'],
                'composite_score': results['composite'],
                'turbulence_score': results['turbulence'],
                'spectral_slope': slope_val,
            }
            # Always expose features so batch.py can collect them without re-computing
            results['_features'] = features

            if self._enable_models:
                self._sequence_manager.add_features(features)
                model_results = self._run_all_models(features)
                results['models'] = model_results
                results['ensemble_probability'] = self._calculate_ensemble(model_results)

                if 'fft_physics' in model_results:
                    results['ml_probability'] = model_results['fft_physics'].get('probability', 0.0)
                else:
                    results['ml_probability'] = results['ensemble_probability']
            else:
                results['models'] = {}
                results['ensemble_probability'] = 0.0
                results['ml_probability'] = 0.0

        else:
            results['spectral_slope'] = 0.0
            results['light_color'] = "gray"
            results['status_msg'] = "Calibrating..."
            results['ml_probability'] = 0.0
            results['models'] = {}
            results['ensemble_probability'] = 0.0
            results['fft_threshold'] = self.fft_threshold
            results['static_threshold'] = self.critical_threshold
            results['current_sigma'] = self.sigma

        # Store history for trend analysis
        if self.is_calibrated:
            self.trend_history.append((time_sec, static_score))

        return results

    def _run_all_models(self, features: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Run prediction on all active models."""
        results = {}
        for name, model in self.registry.get_active().items():
            try:
                if model.metadata.input_type == InputType.SEQUENCE:
                    buffer = self._sequence_manager.get_buffer(name)
                    if buffer and buffer.is_ready():
                        seq_features = {'sequence': buffer.get_sequence()}
                        result = model.predict(seq_features)
                        results[name] = result.to_dict()
                        results[name]['buffer_status'] = 'ready'
                    else:
                        fill = buffer.fill_ratio() if buffer else 0.0
                        results[name] = {
                            'probability': 0.0,
                            'confidence': 0.0,
                            'buffer_status': f'filling ({fill:.0%})'
                        }
                else:
                    result = model.predict(features)
                    results[name] = result.to_dict()
                # Attach the user-configured trust weight so the ensemble can use it
                results[name]['weight'] = model.metadata.weight
            except Exception as e:
                results[name] = {
                    'probability': 0.0,
                    'confidence': 0.0,
                    'weight': model.metadata.weight,
                    'error': str(e)
                }
        return results

    def _calculate_ensemble(self, model_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate weighted ensemble probability from all model predictions.

        Final weight = confidence × user trust weight, so a user can amplify or
        mute any model's contribution without retraining.
        """
        probabilities = []
        weights = []
        for name, result in model_results.items():
            prob = result.get('probability', 0.0)
            conf = result.get('confidence', 1.0)
            user_weight = result.get('weight', 1.0)
            if 'error' not in result and prob > 0:
                probabilities.append(prob)
                weights.append(conf * user_weight)
        if not probabilities:
            return 0.0
        total_weight = sum(weights)
        if total_weight > 0:
            return sum(p * w for p, w in zip(probabilities, weights)) / total_weight
        return float(np.mean(probabilities))

    def calculate_spectral_slope(self, freqs: np.ndarray, power: np.ndarray) -> float:
        """
        Calculate spectral decay rate (log-log slope).

        Healthy flow: steep slope (~-2.5 to -3.0), Kolmogorov cascade.
        Clogged flow: flat slope (~-1.0 to -1.5), white noise/cavitation.
        """
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
        """Robust linear prediction of time to critical threshold."""
        if len(self.trend_history) < 500:
            return None
        data = list(self.trend_history)
        times = np.array([x[0] for x in data])
        scores = np.array([x[1] for x in data])
        window_len = 50
        if len(scores) > window_len:
            scores_smooth = np.convolve(scores, np.ones(window_len) / window_len, mode='valid')
            times_smooth = times[window_len - 1:]
        else:
            scores_smooth = scores
            times_smooth = times
        X = times_smooth[-1000:].reshape(-1, 1)
        y = scores_smooth[-1000:]
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        r2 = model.score(X, y)
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
        """Calculate time to critical using log-linear regression."""
        self.composite_buffer.append(current_score)
        self.composite_times.append(current_time)
        if len(self.composite_buffer) > 300:
            self.composite_buffer.pop(0)
            self.composite_times.pop(0)
        if len(self.composite_buffer) < 50:
            return "gray", "Initializing..."

        CRITICAL_LIMIT = self.fft_threshold
        WARNING_LEVEL = CRITICAL_LIMIT * 0.2

        if current_score < WARNING_LEVEL:
            return "green", "System Stable"
        try:
            subset_scores = np.array(self.composite_buffer)[-150:]
            subset_times = np.array(self.composite_times)[-150:]
            subset_scores = np.maximum(subset_scores, 1e-9)
            log_scores = np.log(subset_scores)
            slope, intercept = np.polyfit(subset_times, log_scores, 1)
            if slope <= 0:
                return "green", "Stable (No Growth)"
            target_log = np.log(CRITICAL_LIMIT)
            current_log = log_scores[-1]
            seconds_left = (target_log - current_log) / slope
            if seconds_left > 1200:
                return "green", "Slight Trend (>20m)"
            elif seconds_left > 300:
                return "yellow", f"Warning: ~{int(seconds_left / 60)} min left"
            elif seconds_left > 0:
                return "red", f"CRITICAL: < {int(seconds_left)}s"
            else:
                return "red", "FAILURE IMMINENT"
        except Exception:
            return "gray", "Calc Error"