"""
Convex polyhedron geometry class.

This module contains the ConvexPolyhedron class that generates a convex
polyhedron from planes defined by points and normals, and vertex triplets
that define intersections of three planes.
"""

import numpy as np
from typing import List, Tuple, Optional
from geometry.convex_solver import ConvexHullSolver
from export.stl_exporter import STLExporter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull


class ConvexPolyhedron:
    """
    Represents a convex polyhedron built from planes and their intersections.
    
    The polyhedron is defined by:
    - List of plane points (base points for normal vectors)
    - List of plane normal vectors
    - List of triplets of plane indices (each triplet defines a vertex)
    - Convex hull computation from computed vertices
    """
    
    def __init__(
        self,
        plane_points: List[np.ndarray],
        plane_normals: List[np.ndarray],
        vertex_triplets: List[Tuple[int, int, int]],
        tolerance: float = 1e-5
    ):
        """
        Initialize convex polyhedron.
        
        Args:
            plane_points: List of points (base points for normals)
            plane_normals: List of normal vectors to planes
            vertex_triplets: List of tuples (i, j, k) - indices of planes that intersect to form vertices
            tolerance: Tolerance for numerical computations
        """
        # Validate inputs
        if len(plane_points) != len(plane_normals):
            raise ValueError(f"Number of plane points ({len(plane_points)}) must equal number of normals ({len(plane_normals)})")
        
        if len(plane_points) == 0:
            raise ValueError("At least one plane must be provided")
        
        self.plane_points = [np.asarray(p, dtype=float) for p in plane_points]
        self.plane_normals = [np.asarray(n, dtype=float) for n in plane_normals]
        self.vertex_triplets = vertex_triplets
        self.tolerance = tolerance
        
        # Computed data
        self.vertices: Optional[np.ndarray] = None
        self.faces: Optional[List[List[np.ndarray]]] = None
        
        # Initialize solvers
        self.hull_solver = ConvexHullSolver(tolerance=tolerance)
        self.stl_exporter = STLExporter(tolerance=tolerance)
    
    def _compute_plane_equation(
        self, 
        point: np.ndarray, 
        normal: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute plane equation: normal·x + d = 0.
        
        Args:
            point: Point on the plane
            normal: Normal vector to the plane
            
        Returns:
            Tuple (normal, d) where d = -normal·point
        """
        normal = np.asarray(normal, dtype=float)
        point = np.asarray(point, dtype=float)
        d = -np.dot(normal, point)
        return normal, d
    
    def _compute_vertex_intersection(
        self, 
        indices: Tuple[int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Compute intersection point of three planes.
        
        Args:
            indices: Tuple (i, j, k) - indices of three planes
            
        Returns:
            Intersection point or None if planes don't intersect uniquely
        """
        i, j, k = indices
        
        # Validate indices
        if i >= len(self.plane_points) or j >= len(self.plane_points) or k >= len(self.plane_points):
            raise IndexError(f"Plane indices {indices} out of range (max: {len(self.plane_points)-1})")
        
        # Get plane equations
        normal_i, d_i = self._compute_plane_equation(self.plane_points[i], self.plane_normals[i])
        normal_j, d_j = self._compute_plane_equation(self.plane_points[j], self.plane_normals[j])
        normal_k, d_k = self._compute_plane_equation(self.plane_points[k], self.plane_normals[k])
        
        # Build system A*x = b
        A = np.array([normal_i, normal_j, normal_k])
        b = np.array([-d_i, -d_j, -d_k])
        
        # Check condition number to avoid singular matrices
        try:
            cond_num = np.linalg.cond(A)
            if cond_num > 1e12:  # Matrix is nearly singular
                return None
            
            # Solve system
            vertex = np.linalg.solve(A, b)
            return vertex
            
        except np.linalg.LinAlgError:
            # System is singular or has no solution
            return None
    
    def _validate_intersection(
        self, 
        indices: Tuple[int, int, int], 
        vertex: np.ndarray
    ) -> bool:
        """
        Validate that vertex lies on all three planes.
        
        Args:
            indices: Tuple (i, j, k) - indices of three planes
            vertex: Computed intersection point
            
        Returns:
            True if vertex is valid (lies on all three planes within tolerance)
        """
        i, j, k = indices
        
        # Compute plane equations
        normal_i, d_i = self._compute_plane_equation(self.plane_points[i], self.plane_normals[i])
        normal_j, d_j = self._compute_plane_equation(self.plane_points[j], self.plane_normals[j])
        normal_k, d_k = self._compute_plane_equation(self.plane_points[k], self.plane_normals[k])
        
        # Check distance from vertex to each plane
        dist_i = abs(np.dot(normal_i, vertex) + d_i)
        dist_j = abs(np.dot(normal_j, vertex) + d_j)
        dist_k = abs(np.dot(normal_k, vertex) + d_k)
        
        # All distances should be within tolerance
        return (dist_i < self.tolerance and 
                dist_j < self.tolerance and 
                dist_k < self.tolerance)
    
    def build_vertices(self) -> np.ndarray:
        """
        Build all vertices from plane triplets.
        
        Returns:
            Array of vertices (N x 3)
        """
        vertices = []
        
        for triplet in self.vertex_triplets:
            vertex = self._compute_vertex_intersection(triplet)
            
            if vertex is not None:
                # Validate intersection
                if self._validate_intersection(triplet, vertex):
                    vertices.append(vertex)
                else:
                    print(f"Warning: Vertex {triplet} failed validation")
            else:
                print(f"Warning: Could not compute vertex for triplet {triplet}")
        
        if len(vertices) == 0:
            raise ValueError("No valid vertices could be computed")
        
        self.vertices = np.array(vertices)
        return self.vertices
    
    def build_convex_hull(self) -> List[List[np.ndarray]]:
        """
        Build convex hull from computed vertices.
        
        Returns:
            List of triangular faces
        """
        if self.vertices is None:
            self.build_vertices()
        
        if len(self.vertices) < 4:
            raise ValueError(f"Not enough vertices ({len(self.vertices)}) for 3D convex hull (minimum: 4)")
        
        # Use ConvexHullSolver to build hull
        self.faces = self.hull_solver.build_convex_hull(self.vertices)
        
        if len(self.faces) == 0:
            raise ValueError("Convex hull construction failed")
        
        return self.faces
    
    def get_faces(self) -> List[List[np.ndarray]]:
        """
        Get triangular faces (builds convex hull if needed).
        
        Returns:
            List of triangular faces
        """
        if self.faces is None:
            self.build_convex_hull()
        
        return self.faces
    
    def to_stl(
        self, 
        filename: str, 
        solid_name: str = "polyhedron", 
        format: str = "ascii"
    ) -> None:
        """
        Export polyhedron to STL file.
        
        Args:
            filename: Output filename
            solid_name: Name of solid in STL file
            format: "ascii" or "binary"
        """
        # Get faces
        faces = self.get_faces()
        
        # Convert to format expected by STLExporter: List of blocks (here single block)
        blocks = [faces]
        
        # Export
        if format.lower() == "ascii":
            self.stl_exporter.write_ascii_stl(blocks, filename, solid_name)
        elif format.lower() == "binary":
            self.stl_exporter.write_binary_stl(blocks, filename, solid_name)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'ascii' or 'binary'")
    
    def visualize(
        self,
        ax: Optional[Axes3D] = None,
        show_planes: bool = True,
        show_vertices: bool = True,
        show_polyhedron: bool = True,
        show_edges: bool = True,
        scale_normal: float = 1.0
    ) -> Axes3D:
        """
        Visualize polyhedron in 3D.
        
        Args:
            ax: Existing 3D axis (creates new if None)
            show_planes: Show plane points and normal vectors
            show_vertices: Show computed vertices
            show_polyhedron: Show convex hull polyhedron (triangular faces)
            show_edges: Show external edges of polyhedron (not internal triangulation)
            scale_normal: Scale factor for normal vectors
            
        Returns:
            3D axis for further customization
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Ensure vertices and faces are built
        if self.vertices is None:
            self.build_vertices()
        if self.faces is None:
            self.build_convex_hull()
        
        # Show plane points (blue dots)
        if show_planes:
            plane_pts_array = np.array(self.plane_points)
            ax.scatter(
                plane_pts_array[:, 0], 
                plane_pts_array[:, 1], 
                plane_pts_array[:, 2],
                c='blue', marker='o', s=50, label='Plane points', alpha=0.7
            )
            
            # Show normal vectors (green arrows)
            for i, (point, normal) in enumerate(zip(self.plane_points, self.plane_normals)):
                normal_scaled = normal * scale_normal
                ax.quiver(
                    point[0], point[1], point[2],
                    normal_scaled[0], normal_scaled[1], normal_scaled[2],
                    color='green', arrow_length_ratio=0.3, alpha=0.6,
                    linewidth=2
                )
        
        # Show computed vertices (orange markers)
        if show_vertices:
            ax.scatter(
                self.vertices[:, 0],
                self.vertices[:, 1],
                self.vertices[:, 2],
                c='orange', marker='*', s=100, label='Vertices', alpha=0.8
            )
        
        # Show convex hull polyhedron (transparent surface)
        if show_polyhedron and self.faces:
            # Create Poly3DCollection from faces
            poly_collection = []
            for face in self.faces:
                poly_collection.append(face)
            
            poly3d = Poly3DCollection(
                poly_collection,
                alpha=0.2,  # Более прозрачные грани, чтобы видеть рёбра
                facecolor='cyan', 
                edgecolor='none',  # Убираем рёбра треугольников, покажем только внешние рёбра
                linewidth=0
            )
            ax.add_collection3d(poly3d)
        
        # Show external edges of polyhedron (not internal triangulation)
        if show_edges and self.vertices is not None and len(self.vertices) >= 4:
            # Build ConvexHull to get edges
            try:
                hull = ConvexHull(self.vertices)
                # Extract unique edges from simplices
                # Each simplex (triangle) has 3 edges, we need to find unique ones
                edges_set = set()
                for simplex in hull.simplices:
                    # Each triangle has 3 edges: (0,1), (1,2), (2,0)
                    # Store as sorted tuples to ensure uniqueness
                    edges_set.add(tuple(sorted([simplex[0], simplex[1]])))
                    edges_set.add(tuple(sorted([simplex[1], simplex[2]])))
                    edges_set.add(tuple(sorted([simplex[2], simplex[0]])))
                
                # Draw edges
                for i, edge_tuple in enumerate(sorted(edges_set)):
                    v1 = self.vertices[edge_tuple[0]]
                    v2 = self.vertices[edge_tuple[1]]
                    # Draw edge as a line
                    ax.plot(
                        [v1[0], v2[0]],
                        [v1[1], v2[1]],
                        [v1[2], v2[2]],
                        color='black',
                        linewidth=1.5,
                        alpha=0.7,
                        label='Edges' if i == 0 else ''  # Label only once
                    )
            except Exception as e:
                print(f"Warning: Could not draw edges: {e}")
        
        # Set labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Convex Polyhedron Visualization')
        
        # Auto-scale view
        if self.vertices is not None:
            all_points = self.vertices
            if show_planes:
                all_points = np.vstack([all_points, np.array(self.plane_points)])
            
            center = np.mean(all_points, axis=0)
            max_dist = np.max(np.linalg.norm(all_points - center, axis=1))
            ax.set_xlim(center[0] - max_dist, center[0] + max_dist)
            ax.set_ylim(center[1] - max_dist, center[1] + max_dist)
            ax.set_zlim(center[2] - max_dist, center[2] + max_dist)
        
        return ax
    
    def __repr__(self) -> str:
        """String representation of polyhedron."""
        num_planes = len(self.plane_points)
        num_vertices = len(self.vertices) if self.vertices is not None else 0
        num_faces = len(self.faces) if self.faces is not None else 0
        return (f"ConvexPolyhedron(planes={num_planes}, "
                f"vertices={num_vertices}, faces={num_faces})")

