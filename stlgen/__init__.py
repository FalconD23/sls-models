"""
STL Layer Generator Package.

This package provides an object-oriented interface for generating
STL models of beveled hexagonal prism layers, replacing the legacy
procedural code with a clean, maintainable architecture.

Main Components:
- GeometryConfig: Configuration for geometric parameters
- CoordinateSystem: Coordinate system transformations
- BeveledPrism: Single beveled hexagonal prism geometry
- PrismLayer: Layer of prisms in hexagonal grid
- ConvexHullSolver: Convex hull computation for faces
- CylindricalTransform: Cylindrical bending transformations
- STLExporter: STL file export functionality
- LayerGenerator: Main facade class

Example Usage:
    from models.stlgen import GeometryConfig, LayerGenerator
    
    config = GeometryConfig(radius=4.0, height=1.0, bev_angle=0.5)
    generator = LayerGenerator(config)
    layer = generator.generate_complete_layer(
        x_num=11, y_num=11, 
        output_filename="output.stl"
    )
"""

from .geometry import (
    GeometryConfig,
    CoordinateSystem,
    BeveledPrism,
    PrismLayer,
    ConvexHullSolver,
    ConvexPolyhedron
)
from .transforms import CylindricalTransform
from .export import STLExporter
from .layer_generator import LayerGenerator

__version__ = "1.0.0"
__author__ = "STL Layer Generator Team"

__all__ = [
    # Geometry classes
    "GeometryConfig",
    "CoordinateSystem", 
    "BeveledPrism",
    "PrismLayer",
    "ConvexHullSolver",
    "ConvexPolyhedron",
    
    # Transform classes
    "CylindricalTransform",
    
    # Export classes
    "STLExporter",
    
    # Main facade
    "LayerGenerator"
]
