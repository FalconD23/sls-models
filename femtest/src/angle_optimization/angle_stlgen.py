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


def generate_stl_for_angle(
    angle_degrees: float,
    output_dir: str = None,
    radius: float = None,
    height: float = None,
    size_trick: float = 1.1,
    x_num: int = 11,
    y_num: int = 11,
    filename_prefix: str = "bev_hex_prisms"
) -> str:
    """
    Generate STL file for given bevel angle.
    
    Args:
        angle_degrees: Bevel angle in degrees
        output_dir: Directory to save STL file (default: script directory)
        radius: Prism radius (default: 4 * (1e2 * 1300 / 7500))
        height: Prism height (default: 1 * 1e2)
        size_trick: Size trick coefficient (default: 1.1)
        x_num: Number of prisms in x direction (default: 11)
        y_num: Number of prisms in y direction (default: 11)
        filename_prefix: Prefix for output filename (default: "bev_hex_prisms")
        
    Returns:
        Path to generated STL file
    """
    # Default parameters
    if radius is None:
        radius = 4 * (1e2 * 1300 / 7500)
    if height is None:
        height = 1 * 1e2
    if output_dir is None:
        output_dir = str(script_dir)
    
    # Convert angle to radians
    bev_angle = np.radians(angle_degrees)
    
    # Create geometry configuration
    config = GeometryConfig(
        radius=radius,
        height=height,
        bev_angle=bev_angle,
        size_trick=size_trick
    )
    
    # Save current working directory
    original_cwd = os.getcwd()
    
    # Change to output directory for STL generation
    os.chdir(output_dir)
    
    try:
        # Create layer generator WITHOUT bending (bending causes QHull errors)
        generator = LayerGenerator(config)
        
        # Generate filename with angle
        filename = f"run_angle_bev_hex_prisms.stl"
        solid_name = f"run_angle_bev_hex_prisms"
        
        # Generate complete layer with all processing steps
        layer = generator.generate_complete_layer(
            x_num=x_num,
            y_num=y_num,
            bend_radius=None,  # No bending
            output_filename=filename,
            solid_name=solid_name,
            format="ascii"
        )
        
        # Return full path to generated STL file
        stl_path = os.path.join(output_dir, filename)
        print(f"Generated STL for angle {angle_degrees:.1f}°: {stl_path} ({len(layer)} prisms)")
        
        return stl_path
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def main():
    """
    Main function demonstrating the new OOP interface.
    
    This replaces the legacy main.py with a clean, object-oriented approach.
    """
    # Generate STL for angle 0 degrees as example
    stl_path = generate_stl_for_angle(
        angle_degrees=0,
        filename_prefix="run_angle_bev_hex_prisms"
    )
    print(f"STL file generated: {stl_path}")



if __name__ == "__main__":
    main()
