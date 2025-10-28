"""
Cylindrical transformation class.

This module contains the CylindricalTransform class that applies
cylindrical bending transformations to prisms and layers.
"""

import numpy as np
from typing import Union, List
from geometry.coordinate_system import CoordinateSystem
from geometry.beveled_prism import BeveledPrism
from geometry.prism_layer import PrismLayer


class CylindricalTransform:
    """
    Applies cylindrical bending transformations.
    
    This class replaces the legacy bev_transform.py functions
    with a more robust and object-oriented approach.
    """
    
    def __init__(
        self,
        bend_radius: float,
        bend_cs: CoordinateSystem
    ):
        """
        Initialize cylindrical transform.
        
        Args:
            bend_radius: Radius of the cylindrical bend
            bend_cs: Coordinate system for the bend
        """
        self.bend_radius = bend_radius
        self.bend_cs = bend_cs
    
    def _local_transform(self, point: np.ndarray, R: float) -> np.ndarray:
        """
        Apply local cylindrical transformation.
        
        This replaces the legacy local_transform function.
        
        Args:
            point: Point in local coordinates
            R: Bend radius
            
        Returns:
            Transformed point
        """
        x0, y0, z0 = point
        
        # Apply cylindrical transformation
        R2 = abs(y0)
        phi_A = x0 / R
        x2 = R2 * np.sin(phi_A)
        y2 = R2 * np.cos(phi_A)
        z2 = z0
        
        return np.array([x2, y2, z2])
    
    def apply_to_point(
        self, 
        point: Union[np.ndarray, List[float]]
    ) -> np.ndarray:
        """
        Apply cylindrical transformation to a point.
        
        This replaces the legacy transform_to_system function.
        
        Args:
            point: Point to transform
            
        Returns:
            Transformed point
        """
        point = np.asarray(point, dtype=float)
        
        # Transform to bend coordinate system
        bend_point = self.bend_cs.transform_point(point, CoordinateSystem.standard())
        
        # Apply local cylindrical transformation
        transformed_point = self._local_transform(bend_point, self.bend_radius)
        
        # Transform back to original coordinate system
        result_point = CoordinateSystem.standard().transform_point(
            transformed_point, self.bend_cs
        )
        
        return result_point
    
    def apply_to_prism(self, prism: BeveledPrism) -> BeveledPrism:
        """
        Apply cylindrical transformation to a prism.
        
        Args:
            prism: BeveledPrism to transform
            
        Returns:
            New transformed BeveledPrism
        """
        # Transform begin and end points
        transformed_begin = np.array([
            self.apply_to_point(point) for point in prism.begin_points
        ])
        transformed_end = np.array([
            self.apply_to_point(point) for point in prism.end_points
        ])
        
        # Create new prism with transformed geometry
        new_prism = BeveledPrism(
            config=prism.config,
            center=self.apply_to_point(prism.center),
            direction_height=prism.direction_height,  # Vectors don't need transformation
            direction_radial=prism.direction_radial
        )
        
        # Override the generated geometry with transformed points
        new_prism._begin_points = transformed_begin
        new_prism._end_points = transformed_end
        new_prism._plane_vectors = [
            transformed_end[i] - transformed_begin[i] 
            for i in range(len(transformed_begin))
        ]
        
        return new_prism
    
    def apply_to_layer(self, layer: PrismLayer) -> PrismLayer:
        """
        Apply cylindrical transformation to a layer.
        
        Args:
            layer: PrismLayer to transform
            
        Returns:
            New transformed PrismLayer
        """
        # Transform each prism in the layer
        transformed_prisms = []
        for prism in layer.prisms:
            transformed_prism = self.apply_to_prism(prism)
            transformed_prisms.append(transformed_prism)
        
        # Create new layer with transformed prisms
        new_layer = PrismLayer(
            config=layer.config,
            layer_cs=layer.layer_cs,
            x_num=layer.x_num,
            y_num=layer.y_num
        )
        
        # Replace prisms with transformed ones
        new_layer._prisms = transformed_prisms
        
        return new_layer
    
    def apply_to_points(
        self, 
        points: Union[np.ndarray, List[List[float]]]
    ) -> np.ndarray:
        """
        Apply cylindrical transformation to multiple points.
        
        Args:
            points: Array of points (N x 3)
            
        Returns:
            Array of transformed points (N x 3)
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        return np.array([
            self.apply_to_point(point) for point in points
        ])
    
    def __repr__(self) -> str:
        """String representation of the transform."""
        return f"CylindricalTransform(bend_radius={self.bend_radius}, bend_cs={self.bend_cs})"
