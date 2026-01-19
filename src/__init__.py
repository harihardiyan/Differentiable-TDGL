"""
Differentiable-TDGL
Core package initialization.

This module exposes the main components of the differentiable TDGL framework:
- TDGL evolution (State, Params, tdgl_step)
- Geometry construction (make_geometry)
- Drift evaluation and optimization utilities
"""

from .tdgl_core import State, Params, tdgl_step
from .geometry import make_geometry
from .optimize import optimize_theta, run_drift

__all__ = [
    "State",
    "Params",
    "tdgl_step",
    "make_geometry",
    "optimize_theta",
    "run_drift",
]
