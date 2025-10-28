"""
Layer generator facade class.

This module contains the LayerGenerator class that orchestrates
the entire STL layer generation process, replacing the legacy main.py.
"""

import numpy as np
from typing import Optional, Callable, List
from geometry.config import GeometryConfig
from geometry.coordinate_system import CoordinateSystem
from geometry.prism_layer import PrismLayer
from geometry.convex_solver import ConvexHullSolver
from transforms.cylindrical import CylindricalTransform
from export.stl_exporter import STLExporter


class LayerGenerator:
    """
    Main facade class for STL layer generation.
    
    This class orchestrates the entire process of generating STL models
    of beveled hexagonal prism layers, replacing the legacy procedural
    main.py with a clean object-oriented interface.
    """
    
    def __init__(
        self,
        config: GeometryConfig,
        bend_radius: Optional[float] = None
    ):
        """
        Initialize layer generator.
        
        Args:
            config: Geometry configuration
            bend_radius: Optional bend radius for cylindrical transformation
        """
        self.config = config
        self.bend_radius = bend_radius
        
        # Initialize components
        self.layer: Optional[PrismLayer] = None
        self.transform: Optional[CylindricalTransform] = None
        self.solver = ConvexHullSolver()
        self.exporter = STLExporter()
        
        # Generated data
        self._faces: Optional[List[List[List[np.ndarray]]]] = None
    
    def generate_layer(
        self,
        x_num: int,
        y_num: int,
        layer_cs: Optional[CoordinateSystem] = None
    ) -> PrismLayer:
        """
        Generate a layer of beveled prisms.
        
        Args:
            x_num: Number of prisms in x direction (half-width)
            y_num: Number of prisms in y direction (half-width)
            layer_cs: Layer coordinate system (default: standard)
            
        Returns:
            Generated PrismLayer
        """
        if layer_cs is None:
            layer_cs = CoordinateSystem.standard()
        
        self.layer = PrismLayer(
            config=self.config,
            layer_cs=layer_cs,
            x_num=x_num,
            y_num=y_num
        )
        
        return self.layer
    
    def apply_bending(
        self,
        bend_radius: Optional[float] = None,
        bend_cs: Optional[CoordinateSystem] = None
    ) -> PrismLayer:
        """
        Apply cylindrical bending to the layer.
        
        Args:
            bend_radius: Bend radius (uses instance default if None)
            bend_cs: Bend coordinate system (default: standard)
            
        Returns:
            Transformed PrismLayer
        """
        if self.layer is None:
            raise ValueError("Layer must be generated before applying bending")
        
        if bend_radius is None:
            bend_radius = self.bend_radius
        
        if bend_radius is None:
            raise ValueError("Bend radius must be specified")
        
        if bend_cs is None:
            # Create bend coordinate system (same as legacy code)
            i_bend = np.array([1, 0, 0])
            j_bend = np.array([0, 0, 1])
            k_bend = np.array([0, -1, 0])
            pivot_bend = np.array([0, 0, bend_radius])
            bend_cs = CoordinateSystem.from_vectors(i_bend, j_bend, k_bend, pivot_bend)
        
        self.transform = CylindricalTransform(bend_radius, bend_cs)
        self.layer = self.transform.apply_to_layer(self.layer)
        
        return self.layer
    
    def compute_faces(
        self,
        connect_condition: Optional[Callable[[np.ndarray], bool]] = None
    ) -> List[List[List[np.ndarray]]]:
        """
        Compute triangular faces for all prisms in the layer.
        
        Args:
            connect_condition: Optional function to filter intersection points
            
        Returns:
            List of face lists, one for each prism
        """
        if self.layer is None:
            raise ValueError("Layer must be generated before computing faces")
        
        # Use default connect condition if none provided
        if connect_condition is None:
            def default_connect_condition(point):
                # Same logic as legacy code
                point = np.asarray(point)
                center = np.array([0, 0, 0])
                axes_vec = np.array([0, 0, 1])
                
                point_rel = point - center
                axes_len = np.linalg.norm(axes_vec)
                axes_ord = axes_vec / axes_len
                rad_dist = np.sqrt(
                    point_rel[0]**2 + point_rel[1]**2 + point_rel[2]**2 - 
                    (np.dot(point_rel, axes_ord)**2)
                )
                
                height = self.config.height
                radius = self.config.radius
                bev_angle = self.config.bev_angle
                kostyl_border = 1  # Legacy "костыль"
                
                if (point[2]**2 > height**2 or 
                    rad_dist > (radius + 0.5 * height / np.tan(bev_angle)) * kostyl_border):
                    return True
                return True
        
        self._faces = self.solver.solve_for_layer(
            self.layer.prisms, 
            connect_condition
        )
        
        return self._faces
    
    def export_stl(
        self,
        filename: str,
        solid_name: str = "bf8_sls",
        format: str = "ascii"
    ) -> None:
        """
        Export layer to STL file.
        
        Args:
            filename: Output filename
            solid_name: Name of the solid (ASCII only)
            format: Output format ("ascii" or "binary")
        """
        if self._faces is None:
            raise ValueError("Faces must be computed before exporting STL")
        
        self.exporter.write_stl(
            blocks=self._faces,
            filename=filename,
            solid_name=solid_name,
            format=format
        )
    
    def generate_complete_layer(
        self,
        x_num: int,
        y_num: int,
        bend_radius: Optional[float] = None,
        connect_condition: Optional[Callable[[np.ndarray], bool]] = None,
        output_filename: Optional[str] = None,
        solid_name: str = "bf8_sls",
        format: str = "ascii"
    ) -> PrismLayer:
        """
        Generate complete layer with all processing steps.
        
        This is a convenience method that performs all steps in sequence.
        
        Args:
            x_num: Number of prisms in x direction
            y_num: Number of prisms in y direction
            bend_radius: Bend radius for cylindrical transformation
            connect_condition: Optional function to filter intersection points
            output_filename: Optional output filename for STL export
            solid_name: Name of the solid (ASCII only)
            format: Output format ("ascii" or "binary")
            
        Returns:
            Generated PrismLayer
        """
        # Generate layer
        self.generate_layer(x_num, y_num)
        
        # Apply bending if radius specified
        if bend_radius is not None:
            self.apply_bending(bend_radius)
        
        # Compute faces
        self.compute_faces(connect_condition)
        
        # Export STL if filename provided
        if output_filename is not None:
            self.export_stl(output_filename, solid_name, format)
        
        return self.layer
    
    @property
    def faces(self) -> Optional[List[List[List[np.ndarray]]]]:
        """Get computed faces."""
        return self._faces
    
    def __repr__(self) -> str:
        """String representation of the generator."""
        return (
            f"LayerGenerator(config={self.config}, bend_radius={self.bend_radius}, "
            f"layer_generated={self.layer is not None}, faces_computed={self._faces is not None})"
        )
