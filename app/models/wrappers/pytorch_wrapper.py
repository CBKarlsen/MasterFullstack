# app/models/wrappers/pytorch_wrapper.py
"""
Wrapper for PyTorch models (.pt, .pth files).

Supports both full model saves and state_dict saves.
Handles GPU/CPU device management automatically.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from ..base import BaseModel, ModelMetadata, ModelType, InputType, PredictionResult


class PyTorchWrapper(BaseModel):
    """
    Wrapper for PyTorch models.

    Supports:
    - Full model saves: torch.save(model, "model.pt")
    - State dict saves: torch.save(model.state_dict(), "model.pth")
      (requires model class definition in companion .py file or metadata)

    Usage by co-students:
    1. Define and train your PyTorch model (LSTM, CNN, etc.)
    2. Save with: torch.save(model, "models/my_lstm.pt")
    3. Create "models/my_lstm.json" with metadata (required for sequence models)
    4. Model auto-loads when platform starts
    """

    def __init__(self, model: Any, metadata: ModelMetadata, device: str = "cpu"):
        """
        Initialize wrapper with loaded model.

        Args:
            model: PyTorch model instance
            metadata: Model metadata
            device: Device to run inference on ("cpu" or "cuda")
        """
        self._model = model
        self._metadata = metadata
        self._device = device

        # Set model to evaluation mode
        if hasattr(self._model, 'eval'):
            self._model.eval()

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """
        Run prediction using the PyTorch model.

        Args:
            features: Dict with feature names as keys
                For SINGLE: {"static_score": 0.1, ...}
                For SEQUENCE: {"sequence": np.ndarray (seq_len, n_features)}

        Returns:
            PredictionResult with probability
        """
        if not TORCH_AVAILABLE:
            return PredictionResult(probability=0.0, extras={"error": "PyTorch not installed"})

        try:
            with torch.no_grad():
                if self._metadata.input_type == InputType.SEQUENCE:
                    # Sequence input
                    sequence = features.get("sequence")
                    if sequence is None:
                        return PredictionResult(probability=0.0, confidence=0.0)

                    # Convert to tensor: (batch=1, seq_len, features)
                    X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                else:
                    # Single feature vector
                    vec = [features.get(f, 0.0) for f in self._metadata.input_features]
                    X = torch.tensor([vec], dtype=torch.float32)

                X = X.to(self._device)

                # Forward pass
                output = self._model(X)

                # Convert output to probability
                if output.dim() > 1 and output.size(1) > 1:
                    # Multi-class output, apply softmax
                    probs = torch.softmax(output, dim=1)
                    probability = float(probs[0, 1])  # Class 1 = clogging
                else:
                    # Single output, apply sigmoid
                    probability = float(torch.sigmoid(output).squeeze())

                return PredictionResult(
                    probability=probability,
                    confidence=1.0,
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
            # Check sequence length
            if len(seq) < self._metadata.sequence_length:
                return False
            return True

        return all(f in features for f in self._metadata.input_features)

    def warmup(self) -> None:
        """Run a dummy forward pass to warm up the model."""
        if not TORCH_AVAILABLE:
            return

        try:
            with torch.no_grad():
                if self._metadata.input_type == InputType.SEQUENCE:
                    dummy = torch.zeros(
                        1, self._metadata.sequence_length,
                        len(self._metadata.input_features)
                    ).to(self._device)
                else:
                    dummy = torch.zeros(
                        1, len(self._metadata.input_features)
                    ).to(self._device)

                _ = self._model(dummy)
        except Exception as e:
            print(f"[PyTorchWrapper] Warmup failed: {e}")

    def cleanup(self) -> None:
        """Release GPU memory if applicable."""
        if TORCH_AVAILABLE and self._device != "cpu":
            try:
                del self._model
                torch.cuda.empty_cache()
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
    def load(cls, path: str) -> Optional['PyTorchWrapper']:
        """
        Load PyTorch model from .pt or .pth file.

        Args:
            path: Path to model file

        Returns:
            PyTorchWrapper instance or None if loading fails
        """
        if not TORCH_AVAILABLE:
            print("[PyTorchWrapper] PyTorch not installed")
            return None

        if not os.path.exists(path):
            print(f"[PyTorchWrapper] File not found: {path}")
            return None

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Try loading full model first
            model = torch.load(path, map_location=device, weights_only=False)

            # If it's a state dict, we need the model architecture
            if isinstance(model, dict):
                print(f"[PyTorchWrapper] State dict found, need model architecture for {path}")
                return None

        except Exception as e:
            print(f"[PyTorchWrapper] Failed to load {path}: {e}")
            return None

        # Load metadata
        metadata = cls._load_metadata(path)

        return cls(model=model, metadata=metadata, device=device)

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
                print(f"[PyTorchWrapper] Failed to load metadata: {e}")

        # Generate default metadata
        name = Path(model_path).stem
        return ModelMetadata(
            name=name,
            version="1.0.0",
            author="Unknown",
            description=f"PyTorch model loaded from {Path(model_path).name}",
            model_type=ModelType.PYTORCH,
            input_type=InputType.SINGLE,  # Default, should be overridden in JSON
        )
