# app/models/builtin/isolation_forest.py
"""
Built-in Isolation Forest model — unsupervised anomaly detection.

Unlike the Random Forest (which learns to replicate the physics model's labels),
this model trains ONLY on healthy data and learns the multidimensional distribution
of normal pipe operation.  During inference it measures how far each sample is from
that learned "normal" and converts the distance to a clogging probability.

This gives a genuinely independent second opinion: it can flag anomalies that fall
outside the learned healthy feature space even if the physics threshold hasn't
been crossed yet.
"""

import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..base import BaseModel, ModelMetadata, ModelType, InputType, PredictionResult

_SAVE_DIR = Path(__file__).parent.parent.parent.parent / "models" / "_internal"
_SAVE_PATH = _SAVE_DIR / "isolation_forest.pkl"

_FEATURES = ["static_score", "composite_score", "turbulence_score", "spectral_slope"]


def _synthetic_iso():
    """
    Return an Isolation Forest trained on synthetic healthy data.
    Also returns the (score_min, score_max) percentile range used for normalisation.
    """
    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(42)
    n = 2000

    # Synthetic healthy pipe data — same distribution as RandomForest's healthy class
    healthy = np.column_stack(
        [
            rng.exponential(0.005, n),  # static_score
            rng.exponential(0.015, n),  # composite_score
            rng.exponential(0.010, n),  # turbulence_score
            rng.normal(-2.75, 0.25, n),  # spectral_slope
        ]
    )

    iso = IsolationForest(
        n_estimators=100, contamination=0.05, random_state=42, n_jobs=1
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*sklearn.utils.parallel.*", category=UserWarning
        )
        iso.fit(healthy)
        raw = -iso.score_samples(healthy)  # negate: higher ⟹ more anomalous

    score_min = float(np.percentile(raw, 1))
    score_max = float(np.percentile(raw, 99))
    return iso, score_min, score_max


class IsolationForestModel(BaseModel):
    """
    sklearn IsolationForest — unsupervised anomaly detector.

    Learns the joint distribution of all four features during healthy operation.
    Any sample that lies far outside that distribution gets a high probability,
    regardless of whether any individual threshold has been crossed.

    Input features (in order):
        static_score, composite_score, turbulence_score, spectral_slope
    """

    _FEATURES = _FEATURES

    def __init__(self):
        self._iso = None
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._training_source: str = "synthetic data"
        self._training_stats: Optional[Dict[str, Any]] = None

        # Live training progress — polled by the /training-progress endpoint
        self._training_state: Dict[str, Any] = {
            "phase": "idle",
            "percent": 0,
            "message": "",
        }
        self._training_result: Optional[Dict[str, Any]] = None
        self._training_error: Optional[str] = None

        self._metadata = ModelMetadata(
            name="Isolation Forest",
            version="1.0.0",
            author="Built-in",
            description=(
                "Unsupervised anomaly detector — learns what normal pipe operation looks like "
                "from healthy data only. No labeling needed. Gives a genuinely independent "
                "second opinion by flagging deviations from the learned healthy pattern."
            ),
            model_type=ModelType.BUILTIN,
            input_type=InputType.SINGLE,
            input_features=_FEATURES,
            output_type="probability",
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        try:
            import joblib

            if _SAVE_PATH.exists():
                saved = joblib.load(_SAVE_PATH)
                self._iso = saved["iso"]
                self._score_min = saved.get("score_min", 0.0)
                self._score_max = saved.get("score_max", 1.0)
                self._training_source = saved.get("source", "user data")
                self._training_stats = saved.get("stats")
                print(
                    f"[IsolationForestModel] Loaded saved model ({self._training_source})."
                )
            else:
                self._iso, self._score_min, self._score_max = _synthetic_iso()
                print(
                    "[IsolationForestModel] Trained on synthetic healthy data — ready."
                )
        except ImportError:
            print("[IsolationForestModel] scikit-learn not installed; model disabled.")
            self._metadata.enabled = False
        except Exception as e:
            print(
                f"[IsolationForestModel] Load failed ({e}); falling back to synthetic."
            )
            self._iso, self._score_min, self._score_max = _synthetic_iso()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_healthy: np.ndarray,
        n_estimators: int = 100,
        contamination: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Retrain on healthy-only feature data.

        Args:
            X_healthy: Feature matrix of healthy samples, shape (n_samples, 4).
            n_estimators: Number of isolation trees.
            contamination: Expected fraction of outliers in X_healthy (0.01–0.20).
                           A small value (0.05) accounts for mislabelled healthy points.

        Returns:
            Dict with training stats.
        """
        from sklearn.ensemble import IsolationForest
        import joblib

        iso = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*sklearn.utils.parallel.*", category=UserWarning
            )
            iso.fit(X_healthy)
            raw_scores = -iso.score_samples(
                X_healthy
            )  # negate: higher ⟹ more anomalous
            labels = iso.predict(X_healthy)  # -1 = anomaly, +1 = normal

        score_min = float(np.percentile(raw_scores, 1))
        score_max = float(np.percentile(raw_scores, 99))
        n_flagged = int(np.sum(labels == -1))

        stats: Dict[str, Any] = {
            "n_healthy_samples": int(len(X_healthy)),
            "n_estimators": n_estimators,
            "contamination": contamination,
            "flag_rate_on_training": float(n_flagged / max(len(X_healthy), 1)),
            "score_min": score_min,
            "score_max": score_max,
        }

        self._iso = iso
        self._score_min = score_min
        self._score_max = score_max
        self._training_stats = stats

        _SAVE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "iso": iso,
                "score_min": score_min,
                "score_max": score_max,
                "source": self._training_source,
                "stats": stats,
            },
            _SAVE_PATH,
        )

        return stats

    def set_training_source(self, source: str) -> None:
        self._training_source = source

    def reset_to_synthetic(self) -> None:
        self._iso, self._score_min, self._score_max = _synthetic_iso()
        self._training_source = "synthetic data"
        self._training_stats = None
        if _SAVE_PATH.exists():
            _SAVE_PATH.unlink()
        print("[IsolationForestModel] Reset to synthetic training data.")

    def get_training_info(self) -> Dict[str, Any]:
        return {
            "source": self._training_source,
            "is_user_trained": not self.is_synthetic,
            "is_synthetic": self.is_synthetic,
            "stats": self._training_stats,
        }

    def get_training_state(self) -> Dict[str, Any]:
        return {
            **self._training_state,
            "result": self._training_result,
            "error": self._training_error,
        }

    @property
    def is_synthetic(self) -> bool:
        """True when the model is running on synthetic fallback data (not user-trained)."""
        return self._training_source == "synthetic data"

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _scores_to_probs(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw anomaly scores (higher = more anomalous) to [0, 1]."""
        rng = self._score_max - self._score_min
        if rng < 1e-9:
            return np.zeros(len(raw_scores))
        probs = (raw_scores - self._score_min) / rng
        return np.clip(probs, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_batch(
        self, features_list: List[Dict[str, Any]]
    ) -> List[PredictionResult]:
        """Vectorised batch inference — single score_samples() call for all N rows.

        When the model is on synthetic training data, all predictions return
        probability=0.0 (dormant mode) — the synthetic distribution is not
        calibrated to the user's sensor and would produce meaningless results.
        Users should train the model on their own healthy recordings first.
        """
        dormant = PredictionResult(
            probability=0.0,
            confidence=0.0,
            classification="healthy",
            extras={"synthetic": True, "dormant": True},
        )
        if self._iso is None or not features_list:
            return [dormant] * len(features_list)

        # Return zeros until the user trains on their own healthy data
        if self.is_synthetic:
            return [dormant] * len(features_list)

        X = np.array(
            [[f.get(feat, 0.0) for feat in self._FEATURES] for f in features_list]
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*sklearn.utils.parallel.*", category=UserWarning
            )
            raw_scores = -self._iso.score_samples(X)

        probs = self._scores_to_probs(raw_scores)

        results = []
        for prob in probs:
            p = float(prob)
            confidence = float(abs(p - 0.5) * 2.0)
            if p < 0.3:
                classification = "healthy"
            elif p < 0.65:
                classification = "warning"
            else:
                classification = "critical"
            results.append(
                PredictionResult(
                    probability=p,
                    confidence=confidence,
                    classification=classification,
                )
            )
        return results

    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        dormant = PredictionResult(
            probability=0.0,
            confidence=0.0,
            classification="healthy",
            extras={"synthetic": True, "dormant": True},
        )
        if self._iso is None:
            return dormant

        # Dormant until user trains on their own data
        if self.is_synthetic:
            return dormant

        X = np.array([[features.get(f, 0.0) for f in self._FEATURES]])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*sklearn.utils.parallel.*", category=UserWarning
            )
            raw_score = float(-self._iso.score_samples(X)[0])

        prob = float(self._scores_to_probs(np.array([raw_score]))[0])
        confidence = float(abs(prob - 0.5) * 2.0)

        if prob < 0.3:
            classification = "healthy"
        elif prob < 0.65:
            classification = "warning"
        else:
            classification = "critical"

        return PredictionResult(
            probability=prob, confidence=confidence, classification=classification
        )

    def validate_features(self, features: Dict[str, Any]) -> bool:
        return all(f in features for f in self._FEATURES)
