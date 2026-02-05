# app/models/wrappers/sklearn_wrapper.py
"""
Wrapper for scikit-learn models (.pkl files).

Supports any sklearn estimator with predict_proba() or predict() method.
Optionally reads metadata from a companion .json file.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

from ..base import BaseModel, ModelMetadata, ModelType, InputType, PredictionResult


class SklearnWrapper(BaseModel):
    """
    Wrapper for scikit-learn models.

    Loads .pkl files saved with joblib.dump() and adapts them to BaseModel interface.

    Usage by co-students:
    1. Train your sklearn model (RandomForest, SVM, XGBoost, etc.)
    2. Save with: joblib.dump(model, "models/my_model.pkl")
    3. Optionally create "models/my_model.json" with metadata
    4. Model auto-loads when platform starts
    """

    def __init__(self, model: Any, metadata: ModelMetadata, file_path: str = ""):
        """
        Initialize wrapper with loaded model.

        Args:
            model: sklearn estimator instance
            metadata: Model metadata
            file_path: Original file path
        """
        self._model = model
        self._metadata = metadata
        self._file_path = file_path

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """
        Run prediction using the sklearn model.

        Args:
            features: Dict with feature names as keys
                For SINGLE: {"static_score": 0.1, "composite_score": 0.05, ...}
                For SEQUENCE: {"sequence": np.ndarray (seq_len, n_features)}

        Returns:
            PredictionResult with probability
        """
        if self._metadata.input_type == InputType.SEQUENCE:
            # Sequence input - flatten or use as-is depending on model
            sequence = features.get("sequence")
            if sequence is None:
                return PredictionResult(probability=0.0, confidence=0.0)
            X = sequence.reshape(1, -1)  # Flatten for sklearn
        else:
            # Single feature vector
            X = np.array([[
                features.get(f, 0.0) for f in self._metadata.input_features
            ]])

        try:
            # Try predict_proba first (for classifiers)
            if hasattr(self._model, 'predict_proba'):
                proba = self._model.predict_proba(X)
                # Assume binary classification, class 1 = clogging
                probability = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
            elif hasattr(self._model, 'predict'):
                # Fallback to predict (for regressors)
                pred = self._model.predict(X)
                probability = float(np.clip(pred[0], 0.0, 1.0))
            else:
                return PredictionResult(probability=0.0, confidence=0.0)

            # Get confidence from decision function if available
            confidence = 1.0
            if hasattr(self._model, 'decision_function'):
                try:
                    decision = self._model.decision_function(X)
                    # Convert distance to confidence (sigmoid-like)
                    confidence = float(1.0 / (1.0 + np.exp(-abs(decision[0]))))
                except Exception:
                    pass

            return PredictionResult(
                probability=probability,
                confidence=confidence,
                classification=self._classify(probability),
            )

        except Exception as e:
            return PredictionResult(
                probability=0.0,
                confidence=0.0,
                extras={"error": str(e)}
            )

    def validate_features(self, features: Dict[str, Any]) -> bool:
        """Check if required features are present."""
        if self._metadata.input_type == InputType.SEQUENCE:
            seq = features.get("sequence")
            if seq is None:
                return False
            if not isinstance(seq, np.ndarray):
                return False
            return True

        # For single input, check all required features
        return all(f in features for f in self._metadata.input_features)

    def _classify(self, probability: float) -> str:
        """Convert probability to classification label."""
        if probability < 0.2:
            return "healthy"
        elif probability < 0.7:
            return "warning"
        else:
            return "critical"

    @classmethod
    def load(cls, path: str) -> Optional['SklearnWrapper']:
        """
        Load sklearn model from .pkl file.

        Also looks for companion .json file with same name for metadata.

        Args:
            path: Path to .pkl file

        Returns:
            SklearnWrapper instance or None if loading fails
        """
        if joblib is None:
            print("[SklearnWrapper] joblib not installed")
            return None

        if not os.path.exists(path):
            print(f"[SklearnWrapper] File not found: {path}")
            return None

        try:
            model = joblib.load(path)
        except Exception as e:
            print(f"[SklearnWrapper] Failed to load {path}: {e}")
            return None

        # Load metadata from companion JSON if exists
        metadata = cls._load_metadata(path)

        return cls(model=model, metadata=metadata, file_path=path)

    @classmethod
    def _load_metadata(cls, model_path: str) -> ModelMetadata:
        """Load metadata from companion JSON file or generate defaults."""
        json_path = Path(model_path).with_suffix('.json')

        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                return ModelMetadata.from_dict(data)
            except Exception as e:
                print(f"[SklearnWrapper] Failed to load metadata: {e}")

        # Generate default metadata from filename
        name = Path(model_path).stem
        return ModelMetadata(
            name=name,
            version="1.0.0",
            author="Unknown",
            description=f"sklearn model loaded from {Path(model_path).name}",
            model_type=ModelType.SKLEARN,
            input_type=InputType.SINGLE,
        )
