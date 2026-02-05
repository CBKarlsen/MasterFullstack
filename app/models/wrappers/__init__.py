# app/models/wrappers/__init__.py
"""
Model wrappers for different ML frameworks.

Each wrapper adapts framework-specific model formats to the BaseModel interface.
"""

from .sklearn_wrapper import SklearnWrapper

# Conditional imports for optional frameworks
try:
    from .pytorch_wrapper import PyTorchWrapper
except ImportError:
    PyTorchWrapper = None

try:
    from .tensorflow_wrapper import TensorFlowWrapper
except ImportError:
    TensorFlowWrapper = None

__all__ = ['SklearnWrapper', 'PyTorchWrapper', 'TensorFlowWrapper']
