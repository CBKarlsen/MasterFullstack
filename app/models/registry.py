# app/models/registry.py
"""
Model Registry - Discovery, loading, and management of ML models.

Supports:
- Auto-discovery of models from the models/ directory
- Loading sklearn (.pkl), PyTorch (.pt/.pth), and TensorFlow (.h5) models
- Runtime enable/disable of models
- File watching for hot-reload of new models
"""

import os
import json
import threading
import time
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
from dataclasses import dataclass

from .base import BaseModel, ModelMetadata, ModelType, InputType, PredictionResult


@dataclass
class RegisteredModel:
    """Container for a registered model and its state."""
    model: BaseModel
    file_path: Optional[str] = None
    load_time: float = 0.0
    prediction_count: int = 0
    total_inference_time: float = 0.0
    last_error: Optional[str] = None


class ModelRegistry:
    """
    Central registry for all detection models.

    Handles discovery, loading, and lifecycle management of models.
    Thread-safe for concurrent access.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the registry.

        Args:
            models_dir: Directory to scan for user models.
        """
        self._models: Dict[str, RegisteredModel] = {}
        self._models_dir = Path(models_dir)
        self._watcher_thread: Optional[threading.Thread] = None
        self._watching = False
        self._lock = threading.Lock()

        # Ensure models directory exists
        self._models_dir.mkdir(parents=True, exist_ok=True)

    def discover_models(self) -> List[str]:
        """
        Scan the models directory and load all valid models.

        Returns:
            List of successfully loaded model names.
        """
        loaded = []

        if not self._models_dir.exists():
            print(f"[Registry] Models directory not found: {self._models_dir}")
            return loaded

        # Supported file extensions and their loaders
        extensions = {
            '.pkl': self._load_sklearn_model,
            '.pt': self._load_pytorch_model,
            '.pth': self._load_pytorch_model,
            '.h5': self._load_tensorflow_model,
        }

        for file_path in self._models_dir.iterdir():
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in extensions:
                    try:
                        loader = extensions[suffix]
                        model = loader(str(file_path))
                        if model:
                            self.register(model, str(file_path))
                            loaded.append(model.metadata.name)
                            print(f"[Registry] Loaded: {model.metadata.name} ({suffix})")
                    except Exception as e:
                        print(f"[Registry] Failed to load {file_path.name}: {e}")

        return loaded

    def register(self, model: BaseModel, file_path: Optional[str] = None) -> bool:
        """
        Register a model with the registry.

        Args:
            model: Model instance implementing BaseModel.
            file_path: Optional path to the model file.

        Returns:
            True if registration successful.
        """
        with self._lock:
            name = model.metadata.name

            if name in self._models:
                print(f"[Registry] Warning: Overwriting existing model '{name}'")

            self._models[name] = RegisteredModel(
                model=model,
                file_path=file_path,
                load_time=time.time(),
            )

            # Warm up the model
            try:
                model.warmup()
            except Exception as e:
                print(f"[Registry] Warmup failed for '{name}': {e}")

            return True

    def unregister(self, name: str) -> bool:
        """
        Remove a model from the registry.

        Args:
            name: Model name to remove.

        Returns:
            True if model was removed.
        """
        with self._lock:
            if name in self._models:
                # Cleanup resources
                try:
                    self._models[name].model.cleanup()
                except Exception:
                    pass
                del self._models[name]
                print(f"[Registry] Unregistered: {name}")
                return True
            return False

    def get(self, name: str) -> Optional[BaseModel]:
        """Get a model by name."""
        with self._lock:
            reg = self._models.get(name)
            return reg.model if reg else None

    def get_all(self) -> Dict[str, BaseModel]:
        """Get all registered models."""
        with self._lock:
            return {name: reg.model for name, reg in self._models.items()}

    def get_active(self) -> Dict[str, BaseModel]:
        """Get all enabled models."""
        with self._lock:
            return {
                name: reg.model
                for name, reg in self._models.items()
                if reg.model.metadata.enabled
            }

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a model."""
        with self._lock:
            if name in self._models:
                self._models[name].model.metadata.enabled = enabled
                return True
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """Get metadata for all registered models."""
        with self._lock:
            result = []
            for name, reg in self._models.items():
                meta = reg.model.metadata.to_dict()
                meta['prediction_count'] = reg.prediction_count
                meta['avg_inference_ms'] = (
                    (reg.total_inference_time / reg.prediction_count * 1000)
                    if reg.prediction_count > 0 else 0
                )
                meta['last_error'] = reg.last_error
                result.append(meta)
            return result

    def predict_all(self, features: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Run prediction on all active models.

        Args:
            features: Feature dictionary for prediction.

        Returns:
            Dict mapping model name to prediction result dict.
        """
        results = {}
        active_models = self.get_active()

        for name, model in active_models.items():
            try:
                # Validate features
                if not model.validate_features(features):
                    results[name] = {"error": "Invalid features"}
                    continue

                # Run prediction with timing
                start = time.perf_counter()
                result = model.predict(features)
                elapsed = time.perf_counter() - start

                # Update stats
                with self._lock:
                    if name in self._models:
                        self._models[name].prediction_count += 1
                        self._models[name].total_inference_time += elapsed
                        self._models[name].last_error = None

                results[name] = result.to_dict()

            except Exception as e:
                with self._lock:
                    if name in self._models:
                        self._models[name].last_error = str(e)
                results[name] = {"error": str(e)}

        return results

    # --- Model Loaders ---

    def _load_sklearn_model(self, path: str) -> Optional[BaseModel]:
        """Load a scikit-learn model from .pkl file."""
        from .wrappers.sklearn_wrapper import SklearnWrapper
        return SklearnWrapper.load(path)

    def _load_pytorch_model(self, path: str) -> Optional[BaseModel]:
        """Load a PyTorch model from .pt/.pth file."""
        try:
            from .wrappers.pytorch_wrapper import PyTorchWrapper
            return PyTorchWrapper.load(path)
        except ImportError:
            print("[Registry] PyTorch not installed, skipping .pt/.pth files")
            return None

    def _load_tensorflow_model(self, path: str) -> Optional[BaseModel]:
        """Load a TensorFlow/Keras model from .h5 file."""
        try:
            from .wrappers.tensorflow_wrapper import TensorFlowWrapper
            return TensorFlowWrapper.load(path)
        except ImportError:
            print("[Registry] TensorFlow not installed, skipping .h5 files")
            return None

    # --- File Watching ---

    def start_watching(self, poll_interval: float = 5.0) -> None:
        """
        Start background thread to watch for new model files.

        Args:
            poll_interval: Seconds between directory scans.
        """
        if self._watching:
            return

        self._watching = True
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            args=(poll_interval,),
            daemon=True
        )
        self._watcher_thread.start()
        print(f"[Registry] Started watching {self._models_dir}")

    def stop_watching(self) -> None:
        """Stop the file watcher thread."""
        self._watching = False
        if self._watcher_thread:
            self._watcher_thread.join(timeout=2.0)
            self._watcher_thread = None

    def _watch_loop(self, interval: float) -> None:
        """Background loop to detect new model files."""
        known_files = set()

        # Initialize with current files
        for f in self._models_dir.iterdir():
            if f.is_file():
                known_files.add(f.name)

        while self._watching:
            time.sleep(interval)

            try:
                current_files = {f.name for f in self._models_dir.iterdir() if f.is_file()}

                # Detect new files
                new_files = current_files - known_files
                for filename in new_files:
                    file_path = self._models_dir / filename
                    suffix = file_path.suffix.lower()

                    if suffix in ('.pkl', '.pt', '.pth', '.h5'):
                        print(f"[Registry] Detected new model: {filename}")
                        self.discover_models()  # Re-scan to load new model
                        break

                known_files = current_files

            except Exception as e:
                print(f"[Registry] Watch error: {e}")


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry(models_dir: str = "models") -> ModelRegistry:
    """Get or create the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(models_dir)
    return _registry
