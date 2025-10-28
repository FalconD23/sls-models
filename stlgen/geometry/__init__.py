"""
Geometry module for STL layer generation.

This module contains classes for geometric operations and configurations
used in generating STL models of beveled hexagonal prisms.
"""

from .config import GeometryConfig
from .coordinate_system import CoordinateSystem
from .beveled_prism import BeveledPrism
from .prism_layer import PrismLayer
from .convex_solver import ConvexHullSolver

__all__ = [
    'GeometryConfig',
    'CoordinateSystem', 
    'BeveledPrism',
    'PrismLayer',
    'ConvexHullSolver'
]
