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
from typing import Dict, Any, Optional, List
from sklearn.linear_model import LinearRegression



# ---------------------------------------------------------------------------
# Module-level constants (replaces scattered magic numbers)
# ---------------------------------------------------------------------------

CALIBRATION_STEP_RATIO = 20       # Step = window_size // 20 → ~5% overlap
ADAPTIVE_BASELINE_GUARD = 0.4     # Only adapt baseline when score < 40% of threshold
SPECTRAL_SLOPE_FREQ_MIN = 1.0     # Hz — lower bound for log-log slope fit
SPECTRAL_SLOPE_FREQ_MAX = 50.0    # Hz — upper bound for log-log slope fit
SPECTRAL_SLOPE_MIN_BINS = 5       # Minimum frequency bins required for slope fit
FREQ_SPLIT_MIN_IDX = 2            # Minimum split index into frequency bins
COMPOSITE_BUFFER_MAX_LEN = 300    # Rolling history length for ETA prediction
COMPOSITE_BUFFER_MIN_LEN = 50     # Minimum entries before ETA is computed
ETA_REGRESSION_WINDOW = 150       # Log-linear regression over last N composite values
ETA_STABLE_SECONDS = 1200         # >20 min left → "Slight Trend" (green)
ETA_WARNING_SECONDS = 300         # >5 min left → "Warning" (yellow)
WARNING_FRACTION = 0.7            # Traffic light turns non-green at 70% of threshold
COMPOSITE_BASELINE_FALLBACK = 0.8 # Fallback threshold when no calibration data
FFT_THRESHOLD_FALLBACK = 0.05     # Fallback FFT threshold before calibration


def _sigma_to_pct(sigma: float) -> float:
    """Map sigma to a one-tailed normal percentile, capped at 99.9."""
    return min(99.9, 100.0 * (0.5 * (1.0 + math.erf(sigma / math.sqrt(2.0)))))


def _normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Normalize a power spectrum to unit sum (safe against zero-energy windows)."""
    return spectrum / (np.sum(spectrum) + 1e-10)

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

    def __init__(self, fs: float = 20.0, window_sec: float = 30.0,
                 sigma: float = 3.0, enable_models: bool = True,
                 baseline_adapt_minutes: float = 5.0):
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

        # Adaptive baseline: EMA time constant in samples.
        # alpha = 1 / (fs * 60 * adapt_minutes). Only updates when signal is healthy.
        adapt_samples = self.fs * 60.0 * baseline_adapt_minutes
        self._baseline_alpha = 1.0 / max(adapt_samples, 1.0)

        # Calibration state
        self.is_calibrated = False
        self.baseline_mean = 0.0
        self.baseline_std = 0.0
        self.baseline_spectrum = None
        self.baseline_spectrum_norm = None

        # Baseline composite stats (for percentile-based FFT threshold)
        self.baseline_composite_mean = 0.0
        self.baseline_composite_std = 0.0
        self.baseline_composites: np.ndarray = np.array([])
        self._sorted_composites: np.ndarray = np.array([])  # pre-sorted for O(1) percentile
        self.baseline_static_mean = 0.0
        self.baseline_static_std = 0.0

        # Thresholds (computed during calibration)
        self.critical_threshold = 0.0
        self.fft_threshold = FFT_THRESHOLD_FALLBACK

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
        pct = _sigma_to_pct(self.sigma)

        # Static threshold: sigma * std of raw signal (Gaussian-appropriate)
        self.critical_threshold = self.sigma * self.baseline_std

        # Composite threshold: percentile-based (distribution-free).
        # sigma maps to a one-tailed normal percentile so the UI knob stays intuitive:
        #   sigma=2 → 97.7th pct,  sigma=3 → 99.9th pct,  sigma=4 → 99.997th pct
        if len(self._sorted_composites) > 0:
            self.fft_threshold = float(np.percentile(self._sorted_composites, pct))
        elif self.baseline_composite_mean > 0:
            self.fft_threshold = self.baseline_composite_mean * (1.0 + self.sigma * 0.1)
        else:
            self.fft_threshold = COMPOSITE_BASELINE_FALLBACK

        print(f"Thresholds recalculated (σ={self.sigma:.1f}): "
              f"Static={self.critical_threshold:.5f}, "
              f"Composite={self.fft_threshold:.5f}")

    def calibrate(self, baseline_data: List[float]) -> None:
        """
        Calibrate the detector using baseline (healthy) data.

        Args:
            baseline_data: List of pressure/signal values during normal operation.
        """
        if len(baseline_data) < 2:
            return

        self.baseline_mean = float(np.mean(baseline_data))
        self.baseline_std = float(np.std(baseline_data))
        self.baseline_static_mean = 0.0
        self.baseline_static_std = self.baseline_std

        step = max(1, self.window_size // CALIBRATION_STEP_RATIO)
        freqs = np.fft.rfftfreq(self.window_size, d=1.0 / self.fs)
        spectra_list = self._collect_calibration_windows(baseline_data, step)

        self._calibrate_spectrum(spectra_list, freqs)

        self.is_calibrated = True
        self._recalculate_thresholds()

        print(f"Calibrated! Baseline composite: "
              f"mean={self.baseline_composite_mean:.6f}, "
              f"std={self.baseline_composite_std:.6f}, "
              f"n_windows={len(spectra_list)}")
        print(f"  → fft_threshold (σ={self.sigma}): {self.fft_threshold:.6f}")

    def _collect_calibration_windows(
        self, baseline_data: List[float], step: int
    ) -> list:
        """Slide windows over baseline data and collect FFT spectra."""
        spectra_list = []
        window_size = self.window_size

        for i in range(0, len(baseline_data) - window_size + 1, step):
            segment = np.array(baseline_data[i: i + window_size])
            ham_signal = segment * np.hamming(window_size)
            fft_power = np.abs(np.fft.rfft(ham_signal)) ** 2
            spectra_list.append(fft_power)

        return spectra_list

    def _calibrate_spectrum(self, spectra_list: list, freqs: np.ndarray) -> None:
        """Build baseline FFT fingerprint from calibration windows."""
        if not spectra_list:
            self.baseline_spectrum = None
            self.baseline_spectrum_norm = None
            self.baseline_freqs = freqs
            self.baseline_composites = np.array([])
            self._sorted_composites = np.array([])
            self.baseline_composite_mean = 0.0
            self.baseline_composite_std = 0.0
            return

        spectra_array = np.array(spectra_list)                    # (n_windows, n_bins)

        mean_spectrum = np.mean(spectra_array, axis=0)
        baseline_norm = _normalize_spectrum(mean_spectrum)

        # Vectorised L1 distances on full spectra (DC included)
        spec_sums = np.sum(spectra_array, axis=1, keepdims=True)
        spectra_norm = spectra_array / (spec_sums + 1e-10)
        baseline_composites = np.sum(np.abs(spectra_norm - baseline_norm), axis=1)

        self.baseline_spectrum = mean_spectrum
        self.baseline_spectrum_norm = baseline_norm
        self.baseline_freqs = freqs
        self.baseline_composites = baseline_composites
        self._sorted_composites = np.sort(baseline_composites)    # cached for O(1) percentile
        self.baseline_composite_mean = float(np.mean(baseline_composites))
        self.baseline_composite_std = float(np.std(baseline_composites))

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
        if len(self.buffer) < self.window_size:
            return None

        raw_signal = np.array(self.buffer)
        curr_mean = float(np.mean(raw_signal))
        fft_power, freqs, current_norm, split_idx = self._compute_fft_spectrum(raw_signal)

        results: Dict[str, Any] = {
            'static': self._compute_static_score(curr_mean),
            'composite': self._compute_composite_score(fft_power, current_norm, split_idx),
            'turbulence': self._compute_turbulence_score(raw_signal, curr_mean, fft_power, split_idx),
            'raw_freqs': freqs[freqs > 0].tolist(),
            'raw_spectrum': fft_power[freqs > 0].tolist(),
        }

        if self.is_calibrated:
            self._fill_calibrated_results(results, freqs, fft_power, time_sec)
            self.trend_history.append((time_sec, results['static']))
        else:
            self._fill_uncalibrated_results(results)

        return results

    def _compute_static_score(self, curr_mean: float) -> float:
        """Hydraulic head deviation from baseline mean."""
        if not self.is_calibrated:
            return 0.0
        return abs(curr_mean - self.baseline_mean)

    def _compute_fft_spectrum(
        self, signal: np.ndarray
    ) -> tuple:
        """Compute windowed FFT and split index. Returns (fft_power, freqs, composite_norm, split_idx).

        DC is kept (no detrending) because a constant-Hz pump makes mean pressure
        a direct clogging signature. The turbulence score still detrends separately.
        """
        ham_signal = signal * np.hamming(len(signal))
        fft_power = np.abs(np.fft.rfft(ham_signal)) ** 2
        freqs = np.fft.rfftfreq(len(ham_signal), 1.0 / self.fs)
        composite_norm = _normalize_spectrum(fft_power)
        nyquist = self.fs / 2.0 # Have to divide by 2 
        split_freq = min(1.0, nyquist / 4.0)
        split_idx = max(FREQ_SPLIT_MIN_IDX, int(np.searchsorted(freqs, split_freq)))
        return fft_power, freqs, composite_norm, split_idx

    def _compute_composite_score(
        self, fft_power: np.ndarray, current_norm: np.ndarray, split_idx: int
    ) -> float:
        """L1 spectral distance from baseline shape (calibrated) or high-band fraction (fallback)."""
        if not self.is_calibrated or self.baseline_spectrum_norm is None:
            total_power = float(np.sum(fft_power))
            return float(np.sum(fft_power[split_idx:])) / total_power if total_power > 1e-10 else 0.0

        composite_val = float(np.sum(np.abs(current_norm - self.baseline_spectrum_norm)))
        if composite_val < self.fft_threshold * ADAPTIVE_BASELINE_GUARD:
            self.baseline_spectrum_norm = (
                (1.0 - self._baseline_alpha) * self.baseline_spectrum_norm
                + self._baseline_alpha * current_norm
            )
        return composite_val

    def _compute_turbulence_score(
        self, signal: np.ndarray, curr_mean: float, fft_power: np.ndarray, split_idx: int
    ) -> float:
        """Detrended FFT high-band energy fraction."""
        detrended = (signal - curr_mean) * np.hamming(len(signal))
        fft_pure = np.abs(np.fft.rfft(detrended)) ** 2
        total_pure = float(np.sum(fft_pure[1:]))
        return float(np.sum(fft_pure[split_idx:])) / total_pure if total_pure > 1e-10 else 0.0

    def _fill_calibrated_results(
        self, results: Dict[str, Any], freqs: np.ndarray, fft_power: np.ndarray, time_sec: float
    ) -> None:
        """Populate calibrated-mode fields: slope, traffic light, thresholds, ML."""
        slope_val = self.calculate_spectral_slope(freqs, fft_power)
        results['spectral_slope'] = slope_val

        light_color, status_msg = self.predict_composite_eta(time_sec, results['composite'])
        results['light_color'] = light_color
        results['status_msg'] = status_msg

        results['fft_threshold'] = self.fft_threshold
        results['static_threshold'] = self.critical_threshold
        results['current_sigma'] = self.sigma

        features = {
            'static_score': results['static'],
            'composite_score': results['composite'],
            'turbulence_score': results['turbulence'],
            'spectral_slope': slope_val,
        }
        results['_features'] = features

        if not self._enable_models:
            results.update({'models': {}, 'ensemble_probability': 0.0, 'ml_probability': 0.0})
            return

        self._sequence_manager.add_features(features)
        model_results = self._run_all_models(features)
        results['models'] = model_results
        results['ensemble_probability'] = self._calculate_ensemble(model_results)
        results['ml_probability'] = (
            model_results['fft_physics'].get('probability', 0.0)
            if 'fft_physics' in model_results
            else results['ensemble_probability']
        )

    def _fill_uncalibrated_results(self, results: Dict[str, Any]) -> None:
        """Populate fallback values while calibration is pending."""
        results.update({
            'spectral_slope': 0.0,
            'light_color': 'gray',
            'status_msg': 'Calibrating...',
            'ml_probability': 0.0,
            'models': {},
            'ensemble_probability': 0.0,
            'fft_threshold': self.fft_threshold,
            'static_threshold': self.critical_threshold,
            'current_sigma': self.sigma,
        })

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
        mask = (freqs > SPECTRAL_SLOPE_FREQ_MIN) & (freqs < SPECTRAL_SLOPE_FREQ_MAX)
        if np.sum(mask) < SPECTRAL_SLOPE_MIN_BINS:
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
        trend_min_len = COMPOSITE_BUFFER_MAX_LEN + ETA_REGRESSION_WINDOW
        if len(self.trend_history) < trend_min_len:
            return None
        data = list(self.trend_history)
        times = np.array([x[0] for x in data])
        scores = np.array([x[1] for x in data])
        window_len = COMPOSITE_BUFFER_MIN_LEN
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
        if len(self.composite_buffer) > COMPOSITE_BUFFER_MAX_LEN:
            self.composite_buffer.pop(0)
            self.composite_times.pop(0)
        if len(self.composite_buffer) < COMPOSITE_BUFFER_MIN_LEN:
            return "gray", "Initializing..."

        CRITICAL_LIMIT = self.fft_threshold
        WARNING_LEVEL = CRITICAL_LIMIT * WARNING_FRACTION

        if current_score < WARNING_LEVEL:
            return "green", "System Stable"
        try:
            subset_scores = np.array(self.composite_buffer)[-ETA_REGRESSION_WINDOW:]
            subset_times = np.array(self.composite_times)[-ETA_REGRESSION_WINDOW:]
            subset_scores = np.maximum(subset_scores, 1e-9)
            log_scores = np.log(subset_scores)
            slope, intercept = np.polyfit(subset_times, log_scores, 1)
            if slope <= 0:
                return "green", "Stable (No Growth)"
            target_log = np.log(CRITICAL_LIMIT)
            current_log = log_scores[-1]
            seconds_left = (target_log - current_log) / slope
            if seconds_left > ETA_STABLE_SECONDS:
                return "green", "Slight Trend (>20m)"
            elif seconds_left > ETA_WARNING_SECONDS:
                return "yellow", f"Warning: ~{int(seconds_left / 60)} min left"
            elif seconds_left > 0:
                return "red", f"CRITICAL: < {int(seconds_left)}s"
            else:
                return "red", "FAILURE IMMINENT"
        except Exception:
            return "gray", "Calc Error"