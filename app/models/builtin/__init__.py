# app/models/builtin/__init__.py
"""
Built-in detection models that ship with the platform.

These wrap the existing FFT and physics-based detection methods
in the standard model interface for consistency with ML models.
"""

from .fft_physics import FFTPhysicsModel
from .random_forest import RandomForestModel

__all__ = ['FFTPhysicsModel', 'RandomForestModel']
