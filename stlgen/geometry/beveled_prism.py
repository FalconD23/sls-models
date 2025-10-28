"""
Beveled prism geometry class.

This module contains the BeveledPrism class that generates the geometry
of a single beveled hexagonal prism, combining logic from the legacy
generate_unit_block*.py files.
"""

import numpy as np
from typing import Tuple, List, Optional
from geometry.config import GeometryConfig
from geometry.coordinate_system import CoordinateSystem


class BeveledPrism:
    """
    Represents a beveled hexagonal prism (BF8 - Beveled F8).
    
    This class generates the geometry of a single beveled hexagonal prism
    with 8 vertices: 2 axial vertices (top/bottom) and 6 radial vertices
    forming a hexagonal base with beveled edges.
    
    The prism is defined by:
    - A hexagonal base with 6 vertices
    - Top and bottom axial vertices  
    - Beveled edges connecting the base vertices
    - Configurable orientation and position
    """
    
    def __init__(
        self,
        config: GeometryConfig,
        center: np.ndarray,
        direction_height: np.ndarray,
        direction_radial: np.ndarray,
        local_cs: Optional[CoordinateSystem] = None
    ):
        """
        Initialize beveled prism.
        
        Args:
            config: Geometry configuration parameters
            center: Center point of the prism
            direction_height: Direction vector for height axis
            direction_radial: Direction vector for radial axis
            local_cs: Local coordinate system (optional, will be created if None)
        """
        self.config = config
        self.center = np.asarray(center, dtype=float)
        self.direction_height = np.asarray(direction_height, dtype=float)
        self.direction_radial = np.asarray(direction_radial, dtype=float)
        
        # Normalize direction vectors
        self.direction_height = self.direction_height / np.linalg.norm(self.direction_height)
        self.direction_radial = self.direction_radial / np.linalg.norm(self.direction_radial)
        
        # Create local coordinate system if not provided
        if local_cs is None:
            self.local_cs = self._create_local_coordinate_system()
        else:
            self.local_cs = local_cs
        
        # Initialize geometry
        self._begin_points: Optional[np.ndarray] = None
        self._end_points: Optional[np.ndarray] = None
        self._plane_vectors: Optional[List[np.ndarray]] = None
        self._faces: Optional[List[List[np.ndarray]]] = None
        
        # Generate geometry
        self.generate_geometry()
    
    def _create_local_coordinate_system(self) -> CoordinateSystem:
        """Create local coordinate system for the prism."""
        # Create orthonormal basis
        i = self.direction_radial
        k = self.direction_height
        j = np.cross(k, i)
        j = j / np.linalg.norm(j)
        
        return CoordinateSystem.from_vectors(i, j, k, self.center)
    
    def generate_geometry(self) -> None:
        """
        Generate the prism geometry.
        
        This method replaces the legacy bev_F8_vectors function,
        combining logic from both generate_unit_block*.py files.
        """
        PI = np.pi
        size_trick = self.config.size_trick
        bev_angle = self.config.bev_angle
        height = self.config.height
        face_dist = self.config.face_dist
        
        # Generate begin points (local coordinates)
        begin_points = [np.array([0, 0, height / 2])]
        end_points = [np.array([0, 0, height / 2]) * size_trick]
        
        # Generate 6 radial vertices
        for i in range(6):
            curr_x = face_dist * np.cos(PI / 3 * i)
            curr_y = face_dist * np.sin(PI / 3 * i)
            bev_z = face_dist * (size_trick - 1) * (1 - (i + 1) % 3) * np.sin(bev_angle)
            
            current_point_begin = np.array([curr_x, curr_y, 0])
            begin_points.append(current_point_begin)
            
            current_point_end = current_point_begin * size_trick + np.array([0, 0, bev_z])
            end_points.append(current_point_end)
        
        # Add bottom vertex
        begin_points.append(np.array([0, 0, -height / 2]))
        end_points.append(np.array([0, 0, -height / 2]) * size_trick)
        
        # Transform to global coordinates
        standard_cs = CoordinateSystem.standard()
        self._begin_points = self.local_cs.transform_points(begin_points, standard_cs)
        self._end_points = self.local_cs.transform_points(end_points, standard_cs)
        
        # Generate plane vectors (for intersection calculations)
        self._plane_vectors = [
            self._end_points[i] - self._begin_points[i] 
            for i in range(len(self._begin_points))
        ]
    
    @property
    def begin_points(self) -> np.ndarray:
        """Get begin points of the prism."""
        if self._begin_points is None:
            self.generate_geometry()
        return self._begin_points
    
    @property
    def end_points(self) -> np.ndarray:
        """Get end points of the prism."""
        if self._end_points is None:
            self.generate_geometry()
        return self._end_points
    
    @property
    def plane_vectors(self) -> List[np.ndarray]:
        """Get plane vectors of the prism."""
        if self._plane_vectors is None:
            self.generate_geometry()
        return self._plane_vectors
    
    def get_vertices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get begin and end vertices.
        
        Returns:
            Tuple of (begin_points, end_points) arrays
        """
        return self.begin_points, self.end_points
    
    def get_plane_vectors(self) -> List[np.ndarray]:
        """
        Get plane vectors for intersection calculations.
        
        Returns:
            List of plane vectors
        """
        return self.plane_vectors
    
    def get_faces(self) -> List[List[np.ndarray]]:
        """
        Get triangular faces of the prism.
        
        This method would generate the triangular faces for STL export.
        Currently returns None as face generation is handled by ConvexHullSolver.
        
        Returns:
            List of triangular faces (each face is a list of 3 vertices)
        """
        if self._faces is None:
            # Face generation is handled by ConvexHullSolver
            # This is a placeholder for future implementation
            self._faces = []
        return self._faces
    
    def transform_to_system(self, target_cs: CoordinateSystem) -> 'BeveledPrism':
        """
        Create a new prism transformed to target coordinate system.
        
        Args:
            target_cs: Target coordinate system
            
        Returns:
            New BeveledPrism instance in target coordinate system
        """
        # Transform center and direction vectors
        new_center = self.local_cs.transform_point(self.center, target_cs)
        new_dir_height = self.local_cs.transform_vector(self.direction_height, target_cs)
        new_dir_radial = self.local_cs.transform_vector(self.direction_radial, target_cs)
        
        # Create new prism
        return BeveledPrism(
            config=self.config,
            center=new_center,
            direction_height=new_dir_height,
            direction_radial=new_dir_radial,
            local_cs=target_cs
        )
    
    def __repr__(self) -> str:
        """String representation of the prism."""
        return (
            f"BeveledPrism(config={self.config}, center={self.center}, "
            f"height_dir={self.direction_height}, radial_dir={self.direction_radial})"
        )
