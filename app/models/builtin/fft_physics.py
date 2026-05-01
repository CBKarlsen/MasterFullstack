# app/models/builtin/fft_physics.py
"""
Built-in FFT and Physics-based clogging detection model.

Wraps the existing spectral analysis methods from backend.py in the
standard model interface, allowing it to be compared alongside ML models.
"""

from typing import Dict, Any
import numpy as np

from ..base import BaseModel, ModelMetadata, ModelType, InputType, PredictionResult


class FFTPhysicsModel(BaseModel):
    """
    Physics-based clogging detection using FFT spectral analysis.

    This model uses signal processing techniques to detect anomalies:
    - Spectral slope analysis (Kolmogorov power law)
    - High/Low frequency energy ratio
    - Deviation from calibrated baseline

    Unlike ML models, this doesn't require training - it uses physics
    principles to identify clogging signatures in the frequency domain.
    """

    def __init__(self, fs: float = 20.0, calibrated: bool = False):
        """
        Initialize the FFT physics model.

        Args:
            fs: Sampling frequency in Hz.
            calibrated: Whether baseline calibration has been performed.
        """
        self.fs = fs
        self.calibrated = calibrated

        # Thresholds (tunable)
        self.slope_healthy_min = -3.5
        self.slope_healthy_max = -2.0
        self.ratio_threshold = 0.05

        self._metadata = ModelMetadata(
            name="fft_physics",
            version="1.0.0",
            author="Built-in",
            description="FFT spectral analysis with physics-based anomaly detection",
            model_type=ModelType.BUILTIN,
            input_type=InputType.SINGLE,
            input_features=[
                "static_score",
                "composite_score",
                "spectral_slope",
            ],
            output_type="probability",
            enabled=True,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """
        Calculate clogging probability from physics-based features.

        Uses a multi-factor scoring system:
        1. Spectral slope deviation from healthy range (-2.5 to -3.0)
        2. High-frequency to low-frequency energy ratio
        3. Static deviation from baseline

        Args:
            features: Dict containing static_score, composite_score,
                      turbulence_score, spectral_slope

        Returns:
            PredictionResult with combined probability
        """
        try:
            static = features.get("static_score", 0.0)
            composite = features.get("composite_score", 0.0)
            slope = features.get("spectral_slope", -2.5)

            # Calculate component scores

            # 1. Spectral slope score (0-1)
            # Healthy: steep slope (-2.5 to -3.0)
            # Clogged: flat slope (-1.0 to -1.5)
            slope_score = self._slope_to_score(slope)

            # 2. Composite ratio score (0-1)
            # Higher ratio = more high-frequency energy = turbulence/clogging
            ratio_score = self._ratio_to_score(composite)

            # 3. Static deviation score (0-1)
            # Normalized by typical threshold
            static_score = min(static / 0.1, 1.0) if static > 0 else 0.0

            # Combine scores with weights
            # Slope is most physics-grounded, composite is empirically useful
            weights = {
                "slope": 0.4,
                "ratio": 0.35,
                "static": 0.25,
            }

            probability = (
                weights["slope"] * slope_score
                + weights["ratio"] * ratio_score
                + weights["static"] * static_score
            )

            # Clamp to [0, 1]
            probability = max(0.0, min(1.0, probability))

            # Calculate confidence based on feature consistency
            confidence = self._calculate_confidence(
                slope_score, ratio_score, static_score
            )

            return PredictionResult(
                probability=probability,
                confidence=confidence,
                classification=self._classify(probability),
                extras={
                    "spectral_slope": slope,
                    "slope_score": slope_score,
                    "ratio_score": ratio_score,
                    "static_score": static_score,
                    "traffic_light": self._to_traffic_light(probability),
                },
            )

        except Exception as e:
            return PredictionResult(
                probability=0.0, confidence=0.0, extras={"error": str(e)}
            )

    def validate_features(self, features: Dict[str, Any]) -> bool:
        """Check if required features are present."""
        required = ["static_score", "composite_score", "spectral_slope"]
        return all(f in features for f in required)

    def _slope_to_score(self, slope: float) -> float:
        """
        Convert spectral slope to anomaly score.

        Physics basis: Kolmogorov -5/3 law for turbulence.
        - Healthy laminar flow: steep slope (~-3.0)
        - Turbulent/clogged flow: flat slope (~-1.0)
        """
        if slope == 0:
            return 0.5  # Unknown

        # Map slope from [-3.5, -1.0] to [0, 1]
        # -3.0 -> 0 (healthy)
        # -1.0 -> 1 (clogged)
        normalized = (slope - self.slope_healthy_min) / (
            self.slope_healthy_max - self.slope_healthy_min
        )

        # Invert so higher slope (less negative) = higher score
        score = 1.0 - normalized

        return max(0.0, min(1.0, score))

    def _ratio_to_score(self, ratio: float) -> float:
        """
        Convert high/low frequency ratio to anomaly score.

        Higher ratios indicate more energy in high frequencies,
        which correlates with turbulence and obstruction.
        """
        if ratio <= 0:
            return 0.0

        # Use log scale since ratio can span orders of magnitude
        # Log(0.001) ~ -3, Log(0.1) ~ -1
        log_ratio = np.log10(max(ratio, 1e-10))

        # Map from [-3, 0] to [0, 1]
        score = (log_ratio + 3.0) / 3.0

        return max(0.0, min(1.0, score))

    def _calculate_confidence(
        self, slope_score: float, ratio_score: float, static_score: float
    ) -> float:
        """
        Calculate confidence based on agreement between indicators.

        If all indicators point the same direction, confidence is high.
        If indicators disagree, confidence is lower.
        """
        scores = [slope_score, ratio_score, static_score]
        std = np.std(scores)

        # Lower variance = higher confidence
        # Max std for [0,1] values is 0.5 (half at 0, half at 1)
        confidence = 1.0 - (std / 0.5)

        return max(0.3, min(1.0, confidence))

    def _classify(self, probability: float) -> str:
        """Convert probability to classification label."""
        if probability < 0.2:
            return "healthy"
        elif probability < 0.7:
            return "warning"
        else:
            return "critical"

    def _to_traffic_light(self, probability: float) -> str:
        """Convert probability to traffic light color."""
        if probability < 0.2:
            return "green"
        elif probability < 0.7:
            return "yellow"
        else:
            return "red"
