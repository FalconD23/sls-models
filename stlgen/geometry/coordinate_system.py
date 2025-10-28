"""
Coordinate system class for geometric transformations.

This module contains the CoordinateSystem class that handles
coordinate transformations between different reference frames.
"""

import numpy as np
from typing import Tuple, Union, List
from dataclasses import dataclass


@dataclass
class Basis:
    """Represents an orthonormal basis with origin point."""
    i: np.ndarray  # First basis vector
    j: np.ndarray  # Second basis vector  
    k: np.ndarray  # Third basis vector
    origin: np.ndarray  # Origin point
    
    def __post_init__(self):
        """Validate and normalize basis vectors after initialization."""
        self.i = np.asarray(self.i, dtype=float)
        self.j = np.asarray(self.j, dtype=float)
        self.k = np.asarray(self.k, dtype=float)
        self.origin = np.asarray(self.origin, dtype=float)
        
        # Normalize basis vectors
        self.i = self.i / np.linalg.norm(self.i)
        self.j = self.j / np.linalg.norm(self.j)
        self.k = self.k / np.linalg.norm(self.k)
        
        # Validate orthonormality
        self._validate_orthonormality()
    
    def _validate_orthonormality(self, tolerance: float = 1e-10) -> None:
        """Validate that the basis is orthonormal."""
        # Check that vectors are unit length
        if not np.isclose(np.linalg.norm(self.i), 1.0, atol=tolerance):
            raise ValueError("Basis vector i is not unit length")
        if not np.isclose(np.linalg.norm(self.j), 1.0, atol=tolerance):
            raise ValueError("Basis vector j is not unit length")
        if not np.isclose(np.linalg.norm(self.k), 1.0, atol=tolerance):
            raise ValueError("Basis vector k is not unit length")
        
        # Check orthogonality
        if not np.isclose(np.dot(self.i, self.j), 0.0, atol=tolerance):
            raise ValueError("Basis vectors i and j are not orthogonal")
        if not np.isclose(np.dot(self.i, self.k), 0.0, atol=tolerance):
            raise ValueError("Basis vectors i and k are not orthogonal")
        if not np.isclose(np.dot(self.j, self.k), 0.0, atol=tolerance):
            raise ValueError("Basis vectors j and k are not orthogonal")
        
        # Check right-handedness
        cross_product = np.cross(self.i, self.j)
        if not np.allclose(cross_product, self.k, atol=tolerance):
            raise ValueError("Basis is not right-handed")
    
    def get_transformation_matrix(self) -> np.ndarray:
        """Get the transformation matrix from this basis to standard basis."""
        return np.column_stack((self.i, self.j, self.k))
    
    def get_inverse_transformation_matrix(self) -> np.ndarray:
        """Get the transformation matrix from standard basis to this basis."""
        return np.linalg.inv(self.get_transformation_matrix())


class CoordinateSystem:
    """
    Coordinate system for geometric transformations.
    
    This class handles transformations between different coordinate systems,
    replacing the legacy basis_basis_transform function with a more robust
    and object-oriented approach.
    """
    
    def __init__(self, basis: Basis):
        """
        Initialize coordinate system with given basis.
        
        Args:
            basis: Orthonormal basis defining the coordinate system
        """
        self.basis = basis
    
    @classmethod
    def from_vectors(
        cls,
        i: Union[np.ndarray, List[float]],
        j: Union[np.ndarray, List[float]], 
        k: Union[np.ndarray, List[float]],
        origin: Union[np.ndarray, List[float]] = np.array([0.0, 0.0, 0.0])
    ) -> 'CoordinateSystem':
        """
        Create coordinate system from basis vectors.
        
        Args:
            i: First basis vector
            j: Second basis vector
            k: Third basis vector
            origin: Origin point (default: [0, 0, 0])
            
        Returns:
            New CoordinateSystem instance
        """
        basis = Basis(
            i=np.asarray(i, dtype=float),
            j=np.asarray(j, dtype=float),
            k=np.asarray(k, dtype=float),
            origin=np.asarray(origin, dtype=float)
        )
        return cls(basis)
    
    @classmethod
    def standard(cls) -> 'CoordinateSystem':
        """Create standard coordinate system (identity)."""
        return cls.from_vectors(
            i=[1, 0, 0],
            j=[0, 1, 0], 
            k=[0, 0, 1],
            origin=[0, 0, 0]
        )
    
    def transform_point(
        self, 
        point: Union[np.ndarray, List[float]], 
        target_system: 'CoordinateSystem'
    ) -> np.ndarray:
        """
        Transform a point from this coordinate system to target system.
        
        This replaces the legacy basis_basis_transform function.
        
        Args:
            point: Point coordinates in this system
            target_system: Target coordinate system
            
        Returns:
            Point coordinates in target system
        """
        point = np.asarray(point, dtype=float)
        
        # Transform from this system to global (standard) coordinates
        point_global = self.basis.get_transformation_matrix() @ point + self.basis.origin
        
        # Transform from global to target system
        point_target = target_system.basis.get_inverse_transformation_matrix() @ (
            point_global - target_system.basis.origin
        )
        
        return point_target
    
    def transform_vector(
        self, 
        vector: Union[np.ndarray, List[float]], 
        target_system: 'CoordinateSystem'
    ) -> np.ndarray:
        """
        Transform a vector from this coordinate system to target system.
        
        Args:
            vector: Vector in this system
            target_system: Target coordinate system
            
        Returns:
            Vector in target system
        """
        vector = np.asarray(vector, dtype=float)
        
        # Transform from this system to global (standard) coordinates
        vector_global = self.basis.get_transformation_matrix() @ vector
        
        # Transform from global to target system
        vector_target = target_system.basis.get_inverse_transformation_matrix() @ vector_global
        
        return vector_target
    
    def transform_points(
        self, 
        points: Union[np.ndarray, List[List[float]]], 
        target_system: 'CoordinateSystem'
    ) -> np.ndarray:
        """
        Transform multiple points from this coordinate system to target system.
        
        Args:
            points: Array of points in this system (N x 3)
            target_system: Target coordinate system
            
        Returns:
            Array of points in target system (N x 3)
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        return np.array([
            self.transform_point(point, target_system) 
            for point in points
        ])
    
    def __repr__(self) -> str:
        """String representation of the coordinate system."""
        return (
            f"CoordinateSystem(basis=Basis(i={self.basis.i}, j={self.basis.j}, "
            f"k={self.basis.k}, origin={self.basis.origin}))"
        )
