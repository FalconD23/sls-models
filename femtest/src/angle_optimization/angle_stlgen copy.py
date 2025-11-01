"""
Main entry point for STL layer generation.

This module provides a clean interface for generating STL models
of beveled hexagonal prism layers, replacing the legacy procedural code.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Get the directory where this script is located (for STL file generation)
script_path = Path(__file__).absolute()
script_dir = script_path.parent
current_work_dir = os.getcwd()

# Add stlgen directory to Python path for imports
# Path structure: angle_optimization -> src -> femtest -> sls-models -> stlgen
# Go up 4 levels from angle_optimization to sls-models, then add stlgen
stlgen_dir = script_path.parent.parent.parent.parent / "stlgen"
if not stlgen_dir.exists():
    raise ImportError(f"STLgen directory not found: {stlgen_dir}")
sys.path.insert(0, str(stlgen_dir))

# Import modules from stlgen
from geometry.config import GeometryConfig
from geometry.coordinate_system import CoordinateSystem
from layer_generator import LayerGenerator

# Change to script directory for STL file generation (files will be saved here)
os.chdir(str(script_dir))


def main():
    """
    Main function demonstrating the new OOP interface.
    
    This replaces the legacy main.py with a clean, object-oriented approach.
    """
    # Define geometry parameters (same as legacy code)
    PI = np.pi
    RADIUS = 4 * (1e2 * 1300 / 7500)
    HEIGHT = 1 * 1e2
    BEV_ANGLE = np.radians(0)
    
    # Create geometry configuration
    config = GeometryConfig(
        radius=RADIUS,
        height=HEIGHT,
        bev_angle=BEV_ANGLE,
        size_trick=1.1  #! 1.1 Use the more conservative value from generate_unit_block.py
    )
    
    # Create layer generator WITHOUT bending (bending causes QHull errors)
    generator = LayerGenerator(config)  # No bend_radius
    
    # Generate complete layer with all processing steps
    layer = generator.generate_complete_layer(
        x_num=11,
        y_num=11,
        bend_radius=None,  # No bending, errors in convex hull solver code maybe
        output_filename=f"run_angle_bev_hex_prisms.stl",
        solid_name=f"run_angle_bev_hex_prisms",
        format="ascii"
    )
    
    print(f"Generated layer with {len(layer)} prisms")
    print(f"Configuration: {config}")



if __name__ == "__main__":
    main()
