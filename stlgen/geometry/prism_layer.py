"""
Prism layer class for managing hexagonal grid of beveled prisms.

This module contains the PrismLayer class that manages a layer of beveled
prisms arranged in a hexagonal grid pattern with alternating orientations.
"""

import numpy as np
from typing import List, Tuple, Optional
from geometry.config import GeometryConfig
from geometry.coordinate_system import CoordinateSystem
from geometry.beveled_prism import BeveledPrism


class PrismLayer:
    """
    Manages a layer of beveled prisms in hexagonal grid pattern.
    
    This class replaces the legacy LayerSectionWithOrient_6Edge function
    and manages the generation of a hexagonal grid of beveled prisms
    with alternating orientations.
    """
    
    def __init__(
        self,
        config: GeometryConfig,
        layer_cs: CoordinateSystem,
        x_num: int,
        y_num: int
    ):
        """
        Initialize prism layer.
        
        Args:
            config: Geometry configuration
            layer_cs: Layer coordinate system
            x_num: Number of prisms in x direction (half-width)
            y_num: Number of prisms in y direction (half-width)
        """
        self.config = config
        self.layer_cs = layer_cs
        self.x_num = x_num
        self.y_num = y_num
        
        # Initialize prisms list
        self._prisms: List[BeveledPrism] = []
        self._block_configs: List[Tuple[List[np.ndarray], np.ndarray, float, float]] = []
        
        # Generate the layer
        self.generate_hexagonal_grid()
    
    def get_orientation(self, i: int, j: int) -> Optional[List[np.ndarray]]:
        """
        Get orientation basis for prism at grid position (i, j).
        
        This replaces the legacy DefOrientation function.
        
        Args:
            i: Grid position i
            j: Grid position j
            
        Returns:
            Basis vectors [i, j, k] or None if position should be skipped
        """
        PI = np.pi
        
        # Define basis types (same as legacy code)
        basis_type_A = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
        ]
        basis_type_B = [
            np.array([np.cos(-PI / 3), np.sin(-PI / 3), 0]),
            np.array([np.cos(PI / 6), np.sin(PI / 6), 0]),
            np.array([0, 0, 1]),
        ]
        basis_type_C = [
            -np.array([np.cos(PI / 3), np.sin(PI / 3), 0]),
            -np.array([np.cos(5 * PI / 6), np.sin(5 * PI / 6), 0]),
            np.array([0, 0, 1]),
        ]
        
        # Determine orientation based on grid position
        if i % 2 == 0 and j % 2 == 0:
            return None  # Skip this position
        elif i % 2 == 0 and j % 2 == 1:
            return basis_type_C
        elif i % 2 == 1 and j % 2 == 0:
            return basis_type_A
        elif i % 2 == 1 and j % 2 == 1:
            return basis_type_B
        else:
            return None
    
    def generate_hexagonal_grid(self) -> None:
        """
        Generate hexagonal grid of prisms.
        
        This replaces the legacy LayerSectionWithOrient_6Edge function.
        """
        PI = np.pi
        radius = self.config.radius
        height = self.config.height
        
        # Calculate distance between centers
        c_dist = 2 * radius * np.cos(PI / 6)
        
        # Generate grid positions
        for i in range(-self.x_num, self.x_num + 1):
            for j in range(-self.y_num, self.y_num + 1):
                # Calculate position in hexagonal grid
                curr_x = i * c_dist + j * c_dist * np.cos(PI / 3)
                curr_y = j * c_dist * np.sin(PI / 3)
                
                # Check if position is within radius bounds
                if max(abs(curr_x), abs(curr_y)) <= radius * self.x_num:
                    curr_center = np.array([curr_x, curr_y, 0])
                    
                    # Get orientation for this position
                    block_orient = self.get_orientation(i, j)
                    if block_orient is None:
                        continue  # Skip this position
                    
                    # Transform center to global coordinates
                    global_center = self.layer_cs.transform_point(curr_center, CoordinateSystem.standard())
                    
                    # Transform orientation vectors to global coordinates
                    global_orient = []
                    for vec in block_orient:
                        global_vec = self.layer_cs.transform_vector(vec, CoordinateSystem.standard())
                        global_orient.append(global_vec)
                    
                    # Store block configuration
                    self._block_configs.append((
                        global_orient,
                        global_center,
                        radius,
                        height
                    ))
                    
                    # Create prism
                    prism = BeveledPrism(
                        config=self.config,
                        center=global_center,
                        direction_height=global_orient[2],  # k vector
                        direction_radial=global_orient[0]   # i vector
                    )
                    self._prisms.append(prism)
    
    @property
    def prisms(self) -> List[BeveledPrism]:
        """Get list of prisms in the layer."""
        return self._prisms
    
    @property
    def block_configs(self) -> List[Tuple[List[np.ndarray], np.ndarray, float, float]]:
        """Get block configurations (for compatibility with legacy code)."""
        return self._block_configs
    
    def get_centers(self) -> List[np.ndarray]:
        """Get centers of all prisms in the layer."""
        return [prism.center for prism in self._prisms]
    
    def get_begin_points(self) -> List[np.ndarray]:
        """Get begin points of all prisms."""
        return [prism.begin_points for prism in self._prisms]
    
    def get_end_points(self) -> List[np.ndarray]:
        """Get end points of all prisms."""
        return [prism.end_points for prism in self._prisms]
    
    def get_plane_vectors(self) -> List[List[np.ndarray]]:
        """Get plane vectors of all prisms."""
        return [prism.plane_vectors for prism in self._prisms]
    
    def transform_to_system(self, target_cs: CoordinateSystem) -> 'PrismLayer':
        """
        Create a new layer transformed to target coordinate system.
        
        Args:
            target_cs: Target coordinate system
            
        Returns:
            New PrismLayer instance in target coordinate system
        """
        # Transform layer coordinate system
        new_layer_cs = CoordinateSystem.from_vectors(
            i=self.layer_cs.basis.i,
            j=self.layer_cs.basis.j,
            k=self.layer_cs.basis.k,
            origin=self.layer_cs.basis.origin
        )
        
        # Create new layer
        new_layer = PrismLayer(
            config=self.config,
            layer_cs=new_layer_cs,
            x_num=self.x_num,
            y_num=self.y_num
        )
        
        return new_layer
    
    def __len__(self) -> int:
        """Get number of prisms in the layer."""
        return len(self._prisms)
    
    def __getitem__(self, index: int) -> BeveledPrism:
        """Get prism by index."""
        return self._prisms[index]
    
    def __iter__(self):
        """Iterate over prisms."""
        return iter(self._prisms)
    
    def __repr__(self) -> str:
        """String representation of the layer."""
        return (
            f"PrismLayer(config={self.config}, x_num={self.x_num}, "
            f"y_num={self.y_num}, num_prisms={len(self._prisms)})"
        )
