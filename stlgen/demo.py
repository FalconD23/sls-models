"""
Demonstration script for the new OOP STL layer generator.

This script shows how to use the new object-oriented interface
to generate STL models, replacing the legacy procedural code.
"""

import numpy as np
import os
from .geometry import GeometryConfig, CoordinateSystem
from .layer_generator import LayerGenerator


def demo_basic_usage():
    """Demonstrate basic usage of the new OOP interface."""
    print("=== STL Layer Generator Demo ===")
    print()
    
    # Define geometry parameters (same as legacy code)
    PI = np.pi
    RADIUS = 4 * (1e2 * 1300 / 7500)
    HEIGHT = 1 * 1e2
    BEV_ANGLE = np.radians(30)
    
    print(f"Geometry Parameters:")
    print(f"  Radius: {RADIUS:.3f}")
    print(f"  Height: {HEIGHT:.3f}")
    print(f"  Bevel Angle: {np.degrees(BEV_ANGLE):.1f}°")
    print()
    
    # Create geometry configuration
    config = GeometryConfig(
        radius=RADIUS,
        height=HEIGHT,
        bev_angle=BEV_ANGLE,
        size_trick=1.1
    )
    
    print(f"Configuration: {config}")
    print()
    
    # Create layer generator
    generator = LayerGenerator(config, bend_radius=30 * 1e6)
    
    print("Generating layer...")
    
    # Generate complete layer with all processing steps
    layer = generator.generate_complete_layer(
        x_num=11,
        y_num=11,
        bend_radius=30 * 1e6,
        output_filename="demo_output.stl",
        solid_name="bf8_sls_demo",
        format="ascii"
    )
    
    print(f"✓ Generated layer with {len(layer)} prisms")
    print(f"✓ Computed faces for all prisms")
    print(f"✓ Exported to demo_output.stl")
    print()
    
    # Verify output file
    if os.path.exists("demo_output.stl"):
        file_size = os.path.getsize("demo_output.stl")
        print(f"✓ Output file created: {file_size} bytes")
    else:
        print("✗ Output file not found")
    
    return layer


def demo_step_by_step():
    """Demonstrate step-by-step processing."""
    print("=== Step-by-Step Processing Demo ===")
    print()
    
    # Create configuration
    config = GeometryConfig(
        radius=2.0,
        height=1.0,
        bev_angle=np.radians(30),
        size_trick=1.1
    )
    
    # Create generator
    generator = LayerGenerator(config)
    
    # Step 1: Generate layer
    print("Step 1: Generating layer...")
    layer = generator.generate_layer(x_num=3, y_num=3)
    print(f"  ✓ Generated {len(layer)} prisms")
    
    # Step 2: Apply bending
    print("Step 2: Applying cylindrical bending...")
    bent_layer = generator.apply_bending(bend_radius=10.0)
    print(f"  ✓ Applied bending with radius 10.0")
    
    # Step 3: Compute faces
    print("Step 3: Computing triangular faces...")
    faces = generator.compute_faces()
    print(f"  ✓ Computed faces for {len(faces)} prisms")
    
    # Step 4: Export STL
    print("Step 4: Exporting to STL...")
    generator.export_stl("step_by_step.stl", format="ascii")
    print(f"  ✓ Exported to step_by_step.stl")
    
    print()
    return layer


def demo_coordinate_systems():
    """Demonstrate coordinate system transformations."""
    print("=== Coordinate System Demo ===")
    print()
    
    # Create two coordinate systems
    cs1 = CoordinateSystem.standard()
    cs2 = CoordinateSystem.from_vectors(
        i=[1, 0, 0],
        j=[0, 1, 0],
        k=[0, 0, 1],
        origin=[5, 5, 5]
    )
    
    # Test point transformation
    point = np.array([1, 0, 0])
    transformed = cs1.transform_point(point, cs2)
    
    print(f"Original point: {point}")
    print(f"Transformed point: {transformed}")
    print(f"Expected: [0, -5, -5] (point - origin)")
    print()
    
    # Test with layer generation
    config = GeometryConfig(radius=1.0, height=0.5, bev_angle=np.radians(30))
    generator = LayerGenerator(config)
    
    # Generate layer in custom coordinate system
    layer = generator.generate_layer(x_num=2, y_num=2, layer_cs=cs2)
    print(f"Generated layer in custom coordinate system: {len(layer)} prisms")
    print()


def demo_error_handling():
    """Demonstrate error handling and validation."""
    print("=== Error Handling Demo ===")
    print()
    
    # Test invalid configuration
    try:
        config = GeometryConfig(radius=-1.0, height=1.0, bev_angle=0.5)
        print("✗ Should have failed for negative radius")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test invalid bend radius
    try:
        config = GeometryConfig(radius=1.0, height=1.0, bev_angle=0.5)
        generator = LayerGenerator(config)
        generator.apply_bending(bend_radius=None)
        print("✗ Should have failed for None bend radius")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    print()


def main():
    """Run all demonstrations."""
    print("STL Layer Generator - Object-Oriented Refactoring Demo")
    print("=" * 60)
    print()
    
    # Run demonstrations
    demo_basic_usage()
    demo_step_by_step()
    demo_coordinate_systems()
    demo_error_handling()
    
    print("Demo completed successfully!")
    print()
    print("Key improvements over legacy code:")
    print("  ✓ Object-oriented design with clear separation of concerns")
    print("  ✓ Eliminated code duplication between generate_unit_block*.py")
    print("  ✓ Removed global variables and magic numbers")
    print("  ✓ Added proper error handling and validation")
    print("  ✓ Full type hints for better IDE support")
    print("  ✓ Comprehensive testing suite")
    print("  ✓ Clean, maintainable architecture")


if __name__ == "__main__":
    main()
