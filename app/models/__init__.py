# app/models/__init__.py
"""
Model Plugin System for Pipe Clogging Prediction Platform

This package provides a modular architecture for registering and running
multiple ML models (sklearn, PyTorch, TensorFlow) alongside traditional
FFT-based detection methods.
"""

from .base import BaseModel, ModelMetadata, ModelType, InputType
from .registry import ModelRegistry, get_registry

__all__ = [
    "BaseModel",
    "ModelMetadata",
    "ModelType",
    "InputType",
    "ModelRegistry",
    "get_registry",
]
