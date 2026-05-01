# app/models/wrappers/tensorflow_wrapper.py
"""
Wrapper for TensorFlow/Keras models (.h5 or SavedModel format).

Supports Keras functional and sequential models.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None

from ..base import BaseModel, ModelMetadata, ModelType, InputType, PredictionResult


class TensorFlowWrapper(BaseModel):
    """
    Wrapper for TensorFlow/Keras models.

    Supports:
    - Keras .h5 format: model.save("model.h5")
    - SavedModel format: model.save("model_dir/")

    Usage by co-students:
    1. Build and train your Keras model (Sequential, Functional, etc.)
    2. Save with: model.save("models/my_model.h5")
    3. Create "models/my_model.json" with metadata
    4. Model auto-loads when platform starts
    """

    def __init__(self, model: Any, metadata: ModelMetadata):
        """
        Initialize wrapper with loaded model.

        Args:
            model: Keras model instance
            metadata: Model metadata
        """
        self._model = model
        self._metadata = metadata

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """
        Run prediction using the TensorFlow/Keras model.

        Args:
            features: Dict with feature names as keys
                For SINGLE: {"static_score": 0.1, ...}
                For SEQUENCE: {"sequence": np.ndarray (seq_len, n_features)}

        Returns:
            PredictionResult with probability
        """
        if not TF_AVAILABLE:
            return PredictionResult(
                probability=0.0, extras={"error": "TensorFlow not installed"}
            )

        try:
            if self._metadata.input_type == InputType.SEQUENCE:
                # Sequence input
                sequence = features.get("sequence")
                if sequence is None:
                    return PredictionResult(probability=0.0, confidence=0.0)

                # Shape: (batch=1, seq_len, features)
                X = np.expand_dims(sequence, axis=0).astype(np.float32)
            else:
                # Single feature vector
                vec = [features.get(f, 0.0) for f in self._metadata.input_features]
                X = np.array([vec], dtype=np.float32)

            # Run prediction
            output = self._model.predict(X, verbose=0)

            # Convert output to probability
            if output.shape[-1] > 1:
                # Multi-class output
                probability = float(output[0, 1])  # Class 1 = clogging
            else:
                # Binary output
                probability = float(output[0, 0])

            # Ensure probability is in [0, 1]
            probability = np.clip(probability, 0.0, 1.0)

            return PredictionResult(
                probability=probability,
                confidence=1.0,
                classification=self._classify(probability),
            )

        except Exception as e:
            return PredictionResult(
                probability=0.0, confidence=0.0, extras={"error": str(e)}
            )

    def validate_features(self, features: Dict[str, Any]) -> bool:
        """Check if required features are present."""
        if self._metadata.input_type == InputType.SEQUENCE:
            seq = features.get("sequence")
            if seq is None:
                return False
            if not isinstance(seq, np.ndarray):
                return False
            if len(seq) < self._metadata.sequence_length:
                return False
            return True

        return all(f in features for f in self._metadata.input_features)

    def warmup(self) -> None:
        """Run a dummy forward pass to warm up the model."""
        if not TF_AVAILABLE:
            return

        try:
            if self._metadata.input_type == InputType.SEQUENCE:
                dummy = np.zeros(
                    (
                        1,
                        self._metadata.sequence_length,
                        len(self._metadata.input_features),
                    ),
                    dtype=np.float32,
                )
            else:
                dummy = np.zeros(
                    (1, len(self._metadata.input_features)), dtype=np.float32
                )

            _ = self._model.predict(dummy, verbose=0)
        except Exception as e:
            print(f"[TensorFlowWrapper] Warmup failed: {e}")

    def cleanup(self) -> None:
        """Clear Keras session if needed."""
        if TF_AVAILABLE:
            try:
                keras.backend.clear_session()
            except Exception:
                pass

    def _classify(self, probability: float) -> str:
        """Convert probability to classification label."""
        if probability < 0.2:
            return "healthy"
        elif probability < 0.7:
            return "warning"
        else:
            return "critical"

    @classmethod
    def load(cls, path: str) -> Optional["TensorFlowWrapper"]:
        """
        Load TensorFlow/Keras model from .h5 file or SavedModel directory.

        Args:
            path: Path to model file or directory

        Returns:
            TensorFlowWrapper instance or None if loading fails
        """
        if not TF_AVAILABLE:
            print("[TensorFlowWrapper] TensorFlow not installed")
            return None

        if not os.path.exists(path):
            print(f"[TensorFlowWrapper] File not found: {path}")
            return None

        try:
            # Suppress TF logging during load
            tf.get_logger().setLevel("ERROR")

            if path.endswith(".h5"):
                model = keras.models.load_model(path)
            else:
                # Assume SavedModel directory
                model = keras.models.load_model(path)

        except Exception as e:
            print(f"[TensorFlowWrapper] Failed to load {path}: {e}")
            return None

        # Load metadata
        metadata = cls._load_metadata(path)

        return cls(model=model, metadata=metadata)

    @classmethod
    def _load_metadata(cls, model_path: str) -> ModelMetadata:
        """Load metadata from companion JSON file or generate defaults."""
        # For .h5 files, look for .json with same name
        # For directories, look for metadata.json inside
        if model_path.endswith(".h5"):
            json_path = Path(model_path).with_suffix(".json")
        else:
            json_path = Path(model_path) / "metadata.json"

        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                return ModelMetadata.from_dict(data)
            except Exception as e:
                print(f"[TensorFlowWrapper] Failed to load metadata: {e}")

        # Generate default metadata
        name = Path(model_path).stem
        return ModelMetadata(
            name=name,
            version="1.0.0",
            author="Unknown",
            description=f"TensorFlow model loaded from {Path(model_path).name}",
            model_type=ModelType.TENSORFLOW,
            input_type=InputType.SINGLE,
        )
