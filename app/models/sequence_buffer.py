# app/models/sequence_buffer.py
"""
Sequence Buffer Manager for sliding window models (LSTM, GRU, Transformer, etc.).

Maintains a rolling buffer of feature vectors for models that require
temporal context (sequence input) rather than single feature vectors.
"""

from collections import deque
from typing import Dict, List, Optional
import numpy as np


class SequenceBuffer:
    """
    Maintains a sliding window of feature vectors for sequence models.

    Each sequence model that requires historical context should have its own
    SequenceBuffer instance, configured with the model's required sequence length.

    Example:
        buffer = SequenceBuffer(sequence_length=100, feature_names=["static", "composite"])

        # During simulation loop:
        buffer.add({"static": 0.1, "composite": 0.05})

        if buffer.is_ready():
            sequence = buffer.get_sequence()
            result = model.predict({"sequence": sequence})
    """

    def __init__(self, sequence_length: int, feature_names: List[str]):
        """
        Initialize the sequence buffer.

        Args:
            sequence_length: Number of timesteps required by the model.
            feature_names: List of feature names in order expected by model.
        """
        self.seq_len = sequence_length
        self.features = feature_names
        self.n_features = len(feature_names)
        self.buffer: deque = deque(maxlen=sequence_length)
        self._ready = False

    def add(self, features: Dict[str, float]) -> None:
        """
        Add latest feature vector to the buffer.

        Args:
            features: Dictionary mapping feature names to values.
                      Missing features default to 0.0.
        """
        # Extract features in the correct order
        vec = [features.get(f, 0.0) for f in self.features]
        self.buffer.append(vec)

        # Update ready status
        if len(self.buffer) >= self.seq_len:
            self._ready = True

    def add_vector(self, vector: List[float]) -> None:
        """
        Add a raw feature vector (already in correct order).

        Args:
            vector: List of feature values in the order matching feature_names.
        """
        if len(vector) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(vector)}")
        self.buffer.append(list(vector))
        if len(self.buffer) >= self.seq_len:
            self._ready = True

    def is_ready(self) -> bool:
        """
        Check if buffer has enough samples for prediction.

        Returns:
            True if buffer contains at least sequence_length samples.
        """
        return self._ready

    def get_sequence(self) -> np.ndarray:
        """
        Get the current sequence as a numpy array.

        Returns:
            Array of shape (sequence_length, n_features).

        Raises:
            ValueError if buffer is not ready.
        """
        if not self._ready:
            raise ValueError(
                f"Buffer not ready: {len(self.buffer)}/{self.seq_len} samples"
            )

        return np.array(list(self.buffer), dtype=np.float32)

    def get_latest(self, n: int = 1) -> np.ndarray:
        """
        Get the most recent n feature vectors.

        Args:
            n: Number of recent vectors to return.

        Returns:
            Array of shape (n, n_features).
        """
        n = min(n, len(self.buffer))
        if n == 0:
            return np.zeros((0, self.n_features), dtype=np.float32)

        data = list(self.buffer)[-n:]
        return np.array(data, dtype=np.float32)

    def clear(self) -> None:
        """Clear the buffer and reset ready status."""
        self.buffer.clear()
        self._ready = False

    def fill_count(self) -> int:
        """Return current number of samples in buffer."""
        return len(self.buffer)

    def fill_ratio(self) -> float:
        """Return fill ratio (0.0 to 1.0)."""
        return len(self.buffer) / self.seq_len

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return (
            f"SequenceBuffer(seq_len={self.seq_len}, "
            f"features={self.n_features}, "
            f"fill={len(self.buffer)}/{self.seq_len}, "
            f"ready={self._ready})"
        )


class MultiModelSequenceManager:
    """
    Manages sequence buffers for multiple models simultaneously.

    Creates and maintains separate buffers for each model that requires
    sequence input, based on their metadata.
    """

    def __init__(self):
        """Initialize the manager."""
        self._buffers: Dict[str, SequenceBuffer] = {}

    def register_model(
        self, model_name: str, sequence_length: int, feature_names: List[str]
    ) -> None:
        """
        Register a model that needs sequence buffering.

        Args:
            model_name: Unique identifier for the model.
            sequence_length: Number of timesteps required.
            feature_names: Feature names in expected order.
        """
        self._buffers[model_name] = SequenceBuffer(
            sequence_length=sequence_length, feature_names=feature_names
        )

    def unregister_model(self, model_name: str) -> None:
        """Remove a model's buffer."""
        if model_name in self._buffers:
            del self._buffers[model_name]

    def add_features(self, features: Dict[str, float]) -> None:
        """
        Add features to all registered buffers.

        Args:
            features: Feature dictionary to add to all buffers.
        """
        for buffer in self._buffers.values():
            buffer.add(features)

    def get_buffer(self, model_name: str) -> Optional[SequenceBuffer]:
        """Get buffer for a specific model."""
        return self._buffers.get(model_name)

    def get_ready_sequences(self) -> Dict[str, np.ndarray]:
        """
        Get sequences for all models that have ready buffers.

        Returns:
            Dict mapping model names to their sequence arrays.
        """
        return {
            name: buffer.get_sequence()
            for name, buffer in self._buffers.items()
            if buffer.is_ready()
        }

    def clear_all(self) -> None:
        """Clear all buffers."""
        for buffer in self._buffers.values():
            buffer.clear()

    def get_status(self) -> Dict[str, Dict[str, any]]:
        """Get status of all buffers."""
        return {
            name: {
                "fill_count": buffer.fill_count(),
                "fill_ratio": buffer.fill_ratio(),
                "ready": buffer.is_ready(),
                "seq_length": buffer.seq_len,
            }
            for name, buffer in self._buffers.items()
        }

    def __contains__(self, model_name: str) -> bool:
        return model_name in self._buffers

    def __len__(self) -> int:
        return len(self._buffers)
