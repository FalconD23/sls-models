"""
Convex hull solver for computing prism faces.

This module contains the ConvexHullSolver class that computes
intersection points and builds convex hulls to generate
triangular faces for STL export.
"""

import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Callable, Optional, Tuple
from geometry.beveled_prism import BeveledPrism


class ConvexHullSolver:
    """
    Solver for computing convex hulls and triangular faces.
    
    This class replaces the legacy find_intersection_points and
    plot_convex_hull_and_return_faces functions with a more robust
    and object-oriented approach.
    """
    
    def __init__(self, tolerance: float = 1e-5):
        """
        Initialize convex hull solver.
        
        Args:
            tolerance: Tolerance for numerical computations
        """
        self.tolerance = tolerance
    
    def find_plane_intersections(
        self,
        vectors: List[np.ndarray],
        points: List[np.ndarray],
        connect_condition: Optional[Callable[[np.ndarray], bool]] = None
    ) -> np.ndarray:
        """
        Find intersection points of planes defined by vectors and points.
        
        This replaces the legacy find_intersection_points function.
        
        Args:
            vectors: List of plane normal vectors
            points: List of points on the planes
            connect_condition: Optional function to filter intersection points
            
        Returns:
            Array of intersection points
        """
        if connect_condition is None:
            connect_condition = lambda point: True
        
        # Create plane equations (normal, d) where normal·x + d = 0
        plane_equations = []
        for i in range(len(vectors)):
            vec = vectors[i]
            point = points[i]
            normal = vec
            d = -np.dot(normal, point)
            plane_equations.append((normal, d))
        
        intersection_points = []
        
        # Find intersections of triplets of planes
        # Using the same logic as legacy code: [0,7] with [1,2,3,4,5,6]
        for i in [0, 7]:  # Top and bottom planes
            for j in range(1, 7):  # Side planes
                k = 1 + j % 6  # Next side plane
                
                # Solve system A*x = b where A is 3x3 matrix of normals
                A = np.array([
                    plane_equations[i][0],
                    plane_equations[j][0], 
                    plane_equations[k][0]
                ])
                b = np.array([
                    -plane_equations[i][1],
                    -plane_equations[j][1],
                    -plane_equations[k][1]
                ])
                
                # Check if system is solvable (non-singular)
                try:
                    # Check condition number to avoid singular matrices
                    cond_num = np.linalg.cond(A)
                    if cond_num > 1e12:  # Matrix is nearly singular
                        continue
                    
                    point = np.linalg.solve(A, b)
                    
                    # Apply connection condition filter
                    if connect_condition(point):
                        intersection_points.append(point)
                        
                except np.linalg.LinAlgError:
                    # System is singular or has no solution
                    continue
        
        return np.array(intersection_points) if intersection_points else np.array([]).reshape(0, 3)
    
    def build_convex_hull(self, points: np.ndarray) -> List[List[np.ndarray]]:
        """
        Build convex hull and return triangular faces.
        
        This replaces the legacy plot_convex_hull_and_return_faces function.
        
        Args:
            points: Array of points (N x 3)
            
        Returns:
            List of triangular faces, each face is a list of 3 vertices
        """
        if len(points) < 4:
            # Not enough points for a 3D convex hull
            return []
        
        try:
            # Compute convex hull
            hull = ConvexHull(points)
            
            # Extract triangular faces
            faces = []
            for simplex in hull.simplices:
                # Get vertices of the triangular face
                face = [
                    points[simplex[0]],
                    points[simplex[1]], 
                    points[simplex[2]]
                ]
                faces.append(face)
            
            return faces
            
        except Exception as e:
            # If convex hull fails, return empty list
            print(f"Warning: Convex hull computation failed: {e}")
            return []
    
    def solve_for_prism(
        self,
        prism: BeveledPrism,
        connect_condition: Optional[Callable[[np.ndarray], bool]] = None
    ) -> List[List[np.ndarray]]:
        """
        Solve convex hull for a single prism.
        
        Args:
            prism: BeveledPrism instance
            connect_condition: Optional function to filter intersection points
            
        Returns:
            List of triangular faces
        """
        # Get plane vectors and points
        vectors = prism.get_plane_vectors()
        points = prism.begin_points
        
        # Find intersection points
        intersection_points = self.find_plane_intersections(
            vectors, points, connect_condition
        )
        
        if len(intersection_points) == 0:
            return []
        
        # Build convex hull
        faces = self.build_convex_hull(intersection_points)
        
        return faces
    
    def solve_for_layer(
        self,
        prisms: List[BeveledPrism],
        connect_condition: Optional[Callable[[np.ndarray], bool]] = None
    ) -> List[List[List[np.ndarray]]]:
        """
        Solve convex hull for a layer of prisms.
        
        Args:
            prisms: List of BeveledPrism instances
            connect_condition: Optional function to filter intersection points
            
        Returns:
            List of face lists, one for each prism
        """
        all_faces = []
        
        for prism in prisms:
            faces = self.solve_for_prism(prism, connect_condition)
            all_faces.append(faces)
        
        return all_faces
    
    def __repr__(self) -> str:
        """String representation of the solver."""
        return f"ConvexHullSolver(tolerance={self.tolerance})"
