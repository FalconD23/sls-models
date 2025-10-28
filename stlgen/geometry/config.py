"""
Geometry configuration class for STL layer generation.

This module contains the GeometryConfig class that encapsulates
all geometric parameters used in generating beveled hexagonal prisms.
"""

import numpy as np
from typing import Optional


class GeometryConfig:
    """
    Configuration class for geometric parameters of beveled hexagonal prisms.
    
    This class encapsulates all the geometric parameters needed for generating
    STL models of beveled hexagonal prisms, including dimensions, angles,
    and scaling factors.
    
    Attributes:
        radius (float): Base radius of the hexagonal prism
        height (float): Height of the prism
        bev_angle (float): Bevel angle in radians
        size_trick (float): Scaling factor for geometry generation
        face_dist (float): Distance from center to face (computed from radius)
        bev_vec_comp (float): Bevel vector component (computed from face_dist and bev_angle)
    """
    
    def __init__(
        self,
        radius: float,
        height: float,
        bev_angle: float,
        size_trick: float = 1.1
    ):
        """
        Initialize geometry configuration.
        
        Args:
            radius: Base radius of the hexagonal prism (must be positive)
            height: Height of the prism (must be positive)
            bev_angle: Bevel angle in radians (typically between 0 and π/2)
            size_trick: Scaling factor for geometry generation (default: 1.1)
            
        Raises:
            ValueError: If radius, height, or bev_angle are not positive
        """
        self._validate_parameters(radius, height, bev_angle, size_trick)
        
        self.radius = radius
        self.height = height
        self.bev_angle = bev_angle
        self.size_trick = size_trick
        
        # Computed properties
        self.face_dist = radius * np.cos(np.pi / 6)
        self.bev_vec_comp = self.face_dist * np.sin(bev_angle)
    
    def _validate_parameters(
        self, 
        radius: float, 
        height: float, 
        bev_angle: float, 
        size_trick: float
    ) -> None:
        """Validate input parameters."""
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")
        if height <= 0:
            raise ValueError(f"Height must be positive, got {height}")
        if bev_angle < 0 or bev_angle >= np.pi / 2:
            raise ValueError(f"Bevel angle must be in [0, π/2), got {bev_angle}")
        if size_trick <= 0:
            raise ValueError(f"Size trick must be positive, got {size_trick}")
    
    def to_degrees(self) -> 'GeometryConfig':
        """
        Create a copy with bevel angle in degrees.
        
        Returns:
            New GeometryConfig instance with bev_angle in degrees
        """
        return GeometryConfig(
            radius=self.radius,
            height=self.height,
            bev_angle=np.degrees(self.bev_angle),
            size_trick=self.size_trick
        )
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return (
            f"GeometryConfig(radius={self.radius}, height={self.height}, "
            f"bev_angle={self.bev_angle:.3f}, size_trick={self.size_trick})"
        )
    
    def __eq__(self, other) -> bool:
        """Check equality with another GeometryConfig."""
        if not isinstance(other, GeometryConfig):
            return False
        return (
            np.isclose(self.radius, other.radius) and
            np.isclose(self.height, other.height) and
            np.isclose(self.bev_angle, other.bev_angle) and
            np.isclose(self.size_trick, other.size_trick)
        )
