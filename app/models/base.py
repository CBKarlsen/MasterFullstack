# app/models/base.py
"""
Abstract base classes and data structures for the model plugin system.

All models (sklearn, PyTorch, TensorFlow, custom) must implement the BaseModel
interface to be compatible with the registry and detection pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ModelType(Enum):
    """Supported model frameworks."""

    SKLEARN = "sklearn"  # scikit-learn models (.pkl)
    PYTORCH = "pytorch"  # PyTorch models (.pt, .pth)
    TENSORFLOW = "tensorflow"  # TensorFlow/Keras models (.h5, SavedModel)
    BUILTIN = "builtin"  # Built-in methods (FFT, physics-based)
    CUSTOM = "custom"  # Custom implementations


class InputType(Enum):
    """Model input requirements."""

    SINGLE = "single"  # Single feature vector per prediction
    SEQUENCE = "sequence"  # Sliding window of feature vectors (for LSTM, etc.)


@dataclass
class ModelMetadata:
    """
    Metadata describing a model's properties and requirements.

    This can be provided via a JSON file alongside the model file,
    or auto-generated with defaults.
    """

    name: str
    version: str = "1.0.0"
    author: str = "Unknown"
    description: str = ""
    model_type: ModelType = ModelType.SKLEARN
    input_type: InputType = InputType.SINGLE
    input_features: List[str] = field(
        default_factory=lambda: [
            "static_score",
            "composite_score",
            "turbulence_score",
            "spectral_slope",
        ]
    )
    output_type: str = "probability"  # "probability", "classification", "regression"
    sequence_length: int = 1  # For sequence models: number of timesteps
    framework_version: str = ""  # e.g., "torch==2.0.1"
    enabled: bool = True  # Whether model is active
    weight: float = 1.0  # Ensemble trust weight (0 = ignored, 2 = double influence)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "model_type": self.model_type.value,
            "input_type": self.input_type.value,
            "input_features": self.input_features,
            "output_type": self.output_type,
            "sequence_length": self.sequence_length,
            "framework_version": self.framework_version,
            "enabled": self.enabled,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary (e.g., from JSON file)."""
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "1.0.0"),
            author=data.get("author", "Unknown"),
            description=data.get("description", ""),
            model_type=ModelType(data.get("model_type", "sklearn")),
            input_type=InputType(data.get("input_type", "single")),
            input_features=data.get(
                "input_features",
                [
                    "static_score",
                    "composite_score",
                    "turbulence_score",
                    "spectral_slope",
                ],
            ),
            output_type=data.get("output_type", "probability"),
            sequence_length=data.get("sequence_length", 1),
            framework_version=data.get("framework_version", ""),
            enabled=data.get("enabled", True),
            weight=data.get("weight", 1.0),
        )


@dataclass
class PredictionResult:
    """
    Standardized output from model predictions.

    All models must return at least probability and confidence.
    Additional fields can be included in the extras dict.
    """

    probability: float  # Clogging probability (0.0 - 1.0)
    confidence: float = 1.0  # Model's confidence in prediction (0.0 - 1.0)
    classification: Optional[str] = None  # "healthy", "warning", "critical"
    extras: Dict[str, Any] = field(default_factory=dict)  # Model-specific outputs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "probability": self.probability,
            "confidence": self.confidence,
        }
        if self.classification:
            result["classification"] = self.classification
        result.update(self.extras)
        return result


class BaseModel(ABC):
    """
    Abstract base class that all models must implement.

    This provides a consistent interface for the registry and detection
    pipeline to interact with models regardless of their underlying framework.

    Example usage for co-students:

    ```python
    class MyRandomForest(BaseModel):
        def __init__(self, model_path: str):
            self._model = joblib.load(model_path)
            self._metadata = ModelMetadata(
                name="my_rf_model",
                author="Your Name",
                description="Random Forest trained on pump data"
            )

        @property
        def metadata(self) -> ModelMetadata:
            return self._metadata

        def predict(self, features: Dict[str, Any]) -> PredictionResult:
            X = np.array([[features[f] for f in self.metadata.input_features]])
            prob = self._model.predict_proba(X)[0][1]
            return PredictionResult(probability=prob)

        def validate_features(self, features: Dict[str, Any]) -> bool:
            return all(f in features for f in self.metadata.input_features)
    ```
    """

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return model metadata describing its properties."""
        pass

    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """
        Make a prediction given input features.

        Args:
            features: Dictionary with feature names as keys.
                - For SINGLE input_type: {"static_score": 0.1, ...}
                - For SEQUENCE input_type: {"sequence": np.ndarray (seq_len, n_features)}

        Returns:
            PredictionResult with probability and optional extras.
        """
        pass

    @abstractmethod
    def validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Check if the provided features are valid for this model.

        Args:
            features: Dictionary of feature names and values.

        Returns:
            True if all required features are present and valid.
        """
        pass

    def warmup(self) -> None:
        """
        Optional: Run a dummy inference to warm up the model.

        Useful for deep learning models that have initialization overhead.
        Called once when the model is first loaded.
        """
        pass

    def cleanup(self) -> None:
        """
        Optional: Release resources when model is unloaded.

        Useful for freeing GPU memory or closing file handles.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.metadata.name}' type={self.metadata.model_type.value}>"
