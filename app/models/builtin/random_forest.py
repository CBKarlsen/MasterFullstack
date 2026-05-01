# app/models/builtin/random_forest.py
"""
Built-in Random Forest model for pipe clogging detection.

On first startup the model is trained on synthetic data derived from the known
spectral signatures of healthy vs. clogged pipes.  Once a user trains it on
real data via the /api/models/random_forest/train endpoint the fitted model is
saved to models/_internal/random_forest.pkl so it survives server restarts.
"""

import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..base import BaseModel, ModelMetadata, ModelType, InputType, PredictionResult

# Saved inside models/_internal/ so the main registry watcher never picks it up
_SAVE_DIR = Path(__file__).parent.parent.parent.parent / "models" / "_internal"
_SAVE_PATH = _SAVE_DIR / "random_forest.pkl"


def _synthetic_clf():
    """Return a classifier trained on physics-derived synthetic data."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(42)
    n = 1000

    healthy = np.column_stack(
        [
            rng.exponential(0.005, n),  # static_score
            rng.exponential(0.015, n),  # composite_score
            rng.exponential(0.010, n),  # turbulence_score
            rng.normal(-2.75, 0.25, n),  # spectral_slope
        ]
    )
    clogged = np.column_stack(
        [
            rng.exponential(0.04, n) + 0.02,
            rng.exponential(0.06, n) + 0.05,
            rng.exponential(0.04, n) + 0.02,
            rng.normal(-1.25, 0.30, n),
        ]
    )

    X = np.vstack([healthy, clogged])
    y = np.array([0] * n + [1] * n)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=1
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*sklearn.utils.parallel.*",
            category=UserWarning,
        )
        clf.fit(X, y)
    return clf


class RandomForestModel(BaseModel):
    """
    Scikit-learn RandomForestClassifier — built-in, retrainable via the UI.

    Input features (in order):
        static_score, composite_score, turbulence_score, spectral_slope
    """

    _FEATURES = [
        "static_score",
        "composite_score",
        "turbulence_score",
        "spectral_slope",
    ]

    def __init__(self):
        self._clf = None
        self._training_source: str = "synthetic data"
        self._training_stats: Optional[Dict[str, Any]] = None

        # Live training progress — read by the /training-progress endpoint
        self._training_state: Dict[str, Any] = {
            "phase": "idle",
            "percent": 0,
            "message": "",
        }
        self._training_result: Optional[Dict[str, Any]] = None
        self._training_error: Optional[str] = None

        self._metadata = ModelMetadata(
            name="Random Forest",
            version="1.0.0",
            author="Built-in",
            description=(
                "Random Forest classifier. Train it on your own pipe data "
                "using the Train button below to improve accuracy. "
                "Adjust the Trust level to control how much it influences "
                "the overall clogging score."
            ),
            model_type=ModelType.BUILTIN,
            input_type=InputType.SINGLE,
            input_features=self._FEATURES,
            output_type="probability",
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Load previously saved model, or fall back to synthetic training."""
        try:
            import joblib  # sklearn dependency — always available when sklearn is

            if _SAVE_PATH.exists():
                saved = joblib.load(_SAVE_PATH)
                self._clf = saved["clf"]
                self._training_source = saved.get("source", "user data")
                self._training_stats = saved.get("stats")
                print(
                    f"[RandomForestModel] Loaded saved model ({self._training_source})."
                )
            else:
                self._clf = _synthetic_clf()
                self._training_source = "synthetic data"
                print("[RandomForestModel] Trained on synthetic data — ready.")
        except ImportError:
            print("[RandomForestModel] scikit-learn not installed; model disabled.")
            self._metadata.enabled = False
        except Exception as e:
            print(f"[RandomForestModel] Load failed ({e}); falling back to synthetic.")
            self._clf = _synthetic_clf()
            self._training_source = "synthetic data"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: Optional[int] = 8,
    ) -> Dict[str, Any]:
        """
        Retrain on real-world data extracted from a batch analysis run.

        Args:
            X: Feature matrix, shape (n_samples, 4).
            y: Labels — 0 = healthy, 1 = clogged.
            n_estimators: Number of trees.
            max_depth: Maximum tree depth (None = unlimited).

        Returns:
            Dict with training and hold-out accuracy + sample counts.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import joblib

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*sklearn.utils.parallel.*",
                category=UserWarning,
            )
            clf.fit(X_train, y_train)
            train_acc = float(accuracy_score(y_train, clf.predict(X_train)))
            test_acc = float(accuracy_score(y_test, clf.predict(X_test)))

        importances = {
            f: float(clf.feature_importances_[i]) for i, f in enumerate(self._FEATURES)
        }

        stats: Dict[str, Any] = {
            "n_healthy": int(sum(y == 0)),
            "n_clogged": int(sum(y == 1)),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "feature_importances": importances,
        }

        self._clf = clf
        self._training_stats = stats

        # Persist so the model survives restarts
        _SAVE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"clf": clf, "source": self._training_source, "stats": stats}, _SAVE_PATH
        )

        return stats

    def set_training_source(self, source: str) -> None:
        """Update the human-readable training source label (set before calling train())."""
        self._training_source = source
        # Update the persisted label if a save file exists
        if _SAVE_PATH.exists():
            try:
                import joblib

                saved = joblib.load(_SAVE_PATH)
                saved["source"] = source
                joblib.dump(saved, _SAVE_PATH)
            except Exception:
                pass

    def reset_to_synthetic(self) -> None:
        """Discard user-trained model and revert to synthetic training."""
        self._clf = _synthetic_clf()
        self._training_source = "synthetic data"
        self._training_stats = None
        if _SAVE_PATH.exists():
            _SAVE_PATH.unlink()
        print("[RandomForestModel] Reset to synthetic training data.")

    def get_training_info(self) -> Dict[str, Any]:
        """Return current training status for the UI."""
        return {
            "source": self._training_source,
            "is_user_trained": self._training_source != "synthetic data",
            "stats": self._training_stats,
        }

    def get_training_state(self) -> Dict[str, Any]:
        """Return live progress state polled by the frontend during training."""
        return {
            **self._training_state,
            "result": self._training_result,
            "error": self._training_error,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_batch(
        self, features_list: List[Dict[str, Any]]
    ) -> List[PredictionResult]:
        """
        Vectorized batch inference — call once with N samples instead of N times.
        Sklearn RF handles (N, 4) arrays natively; this is ~100x faster than
        calling predict() in a loop for large datasets.
        """
        if self._clf is None or not features_list:
            return [PredictionResult(probability=0.0, confidence=0.0)] * len(
                features_list
            )

        X = np.array(
            [[f.get(feat, 0.0) for feat in self._FEATURES] for f in features_list]
        )
        importances = self._clf.feature_importances_

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*sklearn.utils.parallel.*",
                category=UserWarning,
            )
            probas = self._clf.predict_proba(X)

        results = []
        for proba in probas:
            prob = float(proba[1])
            confidence = float(abs(prob - 0.5) * 2.0)
            if prob < 0.3:
                classification = "healthy"
            elif prob < 0.65:
                classification = "warning"
            else:
                classification = "critical"
            extras = {f: float(importances[i]) for i, f in enumerate(self._FEATURES)}
            results.append(
                PredictionResult(
                    probability=prob,
                    confidence=confidence,
                    classification=classification,
                    extras={"feature_importances": extras},
                )
            )
        return results

    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        if self._clf is None:
            return PredictionResult(probability=0.0, confidence=0.0)

        X = np.array([[features.get(f, 0.0) for f in self._FEATURES]])
        proba = self._clf.predict_proba(X)[0]
        prob = float(proba[1])

        confidence = float(abs(prob - 0.5) * 2.0)

        if prob < 0.3:
            classification = "healthy"
        elif prob < 0.65:
            classification = "warning"
        else:
            classification = "critical"

        importances = self._clf.feature_importances_
        extras = {f: float(importances[i]) for i, f in enumerate(self._FEATURES)}

        return PredictionResult(
            probability=prob,
            confidence=confidence,
            classification=classification,
            extras={"feature_importances": extras},
        )

    def validate_features(self, features: Dict[str, Any]) -> bool:
        return all(f in features for f in self._FEATURES)
