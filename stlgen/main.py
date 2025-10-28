"""
Main entry point for STL layer generation.

This module provides a clean interface for generating STL models
of beveled hexagonal prism layers, replacing the legacy procedural code.
"""

import numpy as np
import sys
import os

# Add current directory to Python path for direct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import modules directly
from geometry.config import GeometryConfig
from geometry.coordinate_system import CoordinateSystem
from layer_generator import LayerGenerator


def main():
    """
    Main function demonstrating the new OOP interface.
    
    This replaces the legacy main.py with a clean, object-oriented approach.
    """
    # Define geometry parameters (same as legacy code)
    PI = np.pi
    RADIUS = 4 * (1e2 * 1300 / 7500)
    HEIGHT = 1 * 1e2
    BEV_ANGLE = np.radians(40)
    
    # Create geometry configuration
    config = GeometryConfig(
        radius=RADIUS,
        height=HEIGHT,
        bev_angle=BEV_ANGLE,
        size_trick=0.6  #! 1.1 Use the more conservative value from generate_unit_block.py
    )
    
    # Create layer generator WITHOUT bending (bending causes QHull errors)
    generator = LayerGenerator(config)  # No bend_radius
    
    # Generate complete layer with all processing steps
    layer = generator.generate_complete_layer(
        x_num=11,
        y_num=11,
        bend_radius=None,  # No bending, errors in convex hull solver code maybe
        output_filename=f"bf8_ascii_{np.degrees(BEV_ANGLE):.1f}deg_radius_{RADIUS:.1f}mm_height_{HEIGHT:.1f}mm.stl",
        solid_name=f"bf8_{np.degrees(BEV_ANGLE):.1f}deg_radius_{RADIUS:.1f}mm_height_{HEIGHT:.1f}mm",
        format="ascii"
    )
    
    print(f"Generated layer with {len(layer)} prisms")
    print(f"Configuration: {config}")


def create_legacy_compatible_interface():
    """
    Create a legacy-compatible interface for backward compatibility.
    
    This function provides the same interface as the legacy code
    but uses the new OOP implementation internally.
    """
    # This could be used to maintain compatibility with existing code
    # that expects the old function-based interface
    pass


if __name__ == "__main__":
    main()
