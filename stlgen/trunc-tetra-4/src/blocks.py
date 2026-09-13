"""Factories for truncated tetrahedra (primitive + 4 assembly types)."""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import combinations

import numpy as np

_STLGEN = Path(__file__).resolve().parent.parent.parent
if str(_STLGEN) not in sys.path:
    sys.path.insert(0, str(_STLGEN))

from geometry.convex_polyhedron import ConvexPolyhedron

try:
    from .config import BlockKind, Tetra4Config
except ImportError:
    from config import BlockKind, Tetra4Config

# Face vertex indices into (T1, T2, B1, B2)
_TET_FACES = (
    (0, 1, 2),  # F0: T1 T2 B1
    (0, 1, 3),  # F1: T1 T2 B2
    (0, 2, 3),  # F2: T1 B1 B2
    (1, 2, 3),  # F3: T2 B1 B2
)

Z_POS, Z_NEG = 4, 5


def _regular_tet_vertices(a: float) -> np.ndarray:
    h0 = a / (2.0 * np.sqrt(2.0))
    return np.array(
        [
            [-a / 2.0, 0.0, h0],   # T1
            [a / 2.0, 0.0, h0],    # T2
            [0.0, -a / 2.0, -h0],  # B1
            [0.0, a / 2.0, -h0],   # B2
        ],
        dtype=float,
    )


def _outward_plane(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = np.cross(p2 - p1, p3 - p1)
    nlen = np.linalg.norm(normal)
    if nlen < 1e-15:
        raise ValueError("Degenerate tetrahedron face")
    normal = normal / nlen
    point = (p1 + p2 + p3) / 3.0
    # origin is inside the tet → n·point > 0 for outward
    if np.dot(normal, point) < 0:
        normal = -normal
    return point, normal


def create_truncated_tetrahedron(
    a: float,
    half_height: float,
    center: Optional[np.ndarray] = None,
    z_top: Optional[float] = None,
    z_bottom: Optional[float] = None,
    tolerance: float = 1e-5,
) -> Tuple[ConvexPolyhedron, dict]:
    """
    Regular tetrahedron with opposite edges cut by planes parallel to XY.

    Args:
        a: edge length of the uncut tetrahedron
        half_height: default soles at ±half_height (ignored if z_top/z_bottom set)
        center: translation of the whole solid
        z_top / z_bottom: override sole positions in local z (before center)
    """
    if a <= 0 or half_height <= 0:
        raise ValueError("a and half_height must be positive")

    h0 = a / (2.0 * np.sqrt(2.0))
    if z_top is None:
        z_top = half_height
    if z_bottom is None:
        z_bottom = -half_height
    if z_top <= z_bottom:
        raise ValueError(f"z_top ({z_top}) must be > z_bottom ({z_bottom})")
    if z_top >= h0 - 1e-9 or z_bottom <= -h0 + 1e-9:
        raise ValueError(
            f"Cut planes must lie inside the tet (|z| < h0={h0:.6f}), "
            f"got z_top={z_top}, z_bottom={z_bottom}"
        )

    if center is None:
        center = np.zeros(3)
    else:
        center = np.asarray(center, dtype=float)

    verts = _regular_tet_vertices(a)
    plane_points: List[np.ndarray] = []
    plane_normals: List[np.ndarray] = []

    for ia, ib, ic in _TET_FACES:
        pt, n = _outward_plane(verts[ia], verts[ib], verts[ic])
        plane_points.append(pt + center)
        plane_normals.append(n)

    plane_points.append(np.array([0.0, 0.0, z_top]) + center)
    plane_normals.append(np.array([0.0, 0.0, 1.0]))
    plane_points.append(np.array([0.0, 0.0, z_bottom]) + center)
    plane_normals.append(np.array([0.0, 0.0, -1.0]))

    vertex_triplets = [
        (Z_POS, 0, 2),
        (Z_POS, 0, 3),
        (Z_POS, 1, 2),
        (Z_POS, 1, 3),
        (Z_NEG, 0, 2),
        (Z_NEG, 0, 3),
        (Z_NEG, 1, 2),
        (Z_NEG, 1, 3),
    ]

    poly = ConvexPolyhedron(
        plane_points=plane_points,
        plane_normals=plane_normals,
        vertex_triplets=vertex_triplets,
        tolerance=tolerance,
    )
    params = {
        "a": a,
        "half_height": half_height,
        "z_top": z_top,
        "z_bottom": z_bottom,
        "h0": h0,
        "center": center,
    }
    return poly, params


def create_block(
    kind: BlockKind,
    config: Tetra4Config,
    center: Optional[np.ndarray] = None,
    half_sign: int = 1,
) -> ConvexPolyhedron:
    """Build one of the four assembly types from the same primitive."""
    if center is None:
        center = np.zeros(3)

    if kind == BlockKind.DOUBLE_2H:
        poly, _ = create_truncated_tetrahedron(
            a=config.a_double,
            half_height=config.H,
            center=center,
            tolerance=config.tolerance,
        )
    elif kind == BlockKind.HALF_H:
        if half_sign >= 0:
            z_top, z_bottom = config.H, 0.0
        else:
            z_top, z_bottom = 0.0, -config.H
        poly, _ = create_truncated_tetrahedron(
            a=config.a_double,
            half_height=config.H,
            center=center,
            z_top=z_top,
            z_bottom=z_bottom,
            tolerance=config.tolerance,
        )
    elif kind == BlockKind.LARGE_H:
        poly, _ = create_truncated_tetrahedron(
            a=config.a_large,
            half_height=config.H / 2.0,
            center=center,
            tolerance=config.tolerance,
        )
    elif kind == BlockKind.SMALL_H:
        poly, _ = create_truncated_tetrahedron(
            a=config.a_small,
            half_height=config.H / 2.0,
            center=center,
            tolerance=config.tolerance,
        )
    else:
        raise ValueError(f"Unknown block kind: {kind}")

    poly.build_vertices()
    poly.build_convex_hull()
    return poly


def _vertex_inside(plane_points, plane_normals, vertex: np.ndarray, tol: float) -> bool:
    for p, n in zip(plane_points, plane_normals):
        n = np.asarray(n, dtype=float)
        nlen = np.linalg.norm(n)
        if nlen < 1e-15:
            continue
        n = n / nlen
        if np.dot(n, vertex - np.asarray(p, dtype=float)) > tol:
            return False
    return True


def polyhedron_from_planes(
    plane_points: List[np.ndarray],
    plane_normals: List[np.ndarray],
    tolerance: float = 1e-5,
) -> ConvexPolyhedron:
    """Выпуклый полиэдр: все тройки плоскостей, вершины внутри всех полупространств."""
    n = len(plane_points)
    triplets = [(i, j, k) for i, j, k in combinations(range(n), 3)]
    poly = ConvexPolyhedron(
        plane_points=plane_points,
        plane_normals=plane_normals,
        vertex_triplets=triplets,
        tolerance=tolerance,
    )
    raw = []
    for triplet in triplets:
        vertex = poly._compute_vertex_intersection(triplet)
        if vertex is None:
            continue
        if not poly._validate_intersection(triplet, vertex):
            continue
        if _vertex_inside(plane_points, plane_normals, vertex, tolerance * 10):
            raw.append(vertex)
    if len(raw) < 4:
        raise ValueError(f"Need >= 4 interior vertices, got {len(raw)}")
    uniq = []
    for v in raw:
        if not any(np.linalg.norm(v - u) < tolerance * 20 for u in uniq):
            uniq.append(v)
    poly.vertices = np.array(uniq)
    poly.build_convex_hull()
    return poly


def create_white_from_neighbors(
    center: np.ndarray,
    z_lo: float,
    z_hi: float,
    neighbor_polys: List[ConvexPolyhedron],
    tolerance: float = 1e-5,
) -> ConvexPolyhedron:
    """Белый блок высоты H: подошвы + боковые плоскости соседних чёрных (снаружи чёрных)."""
    center = np.asarray(center, dtype=float)
    plane_points: List[np.ndarray] = [
        np.array([center[0], center[1], z_hi]),
        np.array([center[0], center[1], z_lo]),
    ]
    plane_normals: List[np.ndarray] = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    for nb in neighbor_polys:
        for p, n in zip(nb.plane_points, nb.plane_normals):
            n = np.asarray(n, dtype=float)
            nlen = np.linalg.norm(n)
            if nlen < 1e-15:
                continue
            n = n / nlen
            if abs(n[2]) > 0.9:
                continue
            # белый центр снаружи чёрного → эта грань чёрного смотрит на белый
            if np.dot(n, center - np.asarray(p, dtype=float)) <= tolerance:
                continue
            plane_points.append(np.asarray(p, dtype=float).copy())
            plane_normals.append(-n)
    return polyhedron_from_planes(plane_points, plane_normals, tolerance=tolerance)


def translate_polyhedron(polyhedron: ConvexPolyhedron, offset: np.ndarray) -> ConvexPolyhedron:
    offset = np.asarray(offset, dtype=float)
    moved = ConvexPolyhedron(
        plane_points=[pt + offset for pt in polyhedron.plane_points],
        plane_normals=list(polyhedron.plane_normals),
        vertex_triplets=list(polyhedron.vertex_triplets),
        tolerance=polyhedron.tolerance,
    )
    if polyhedron.vertices is not None:
        moved.build_vertices()
    if polyhedron.faces is not None:
        moved.build_convex_hull()
    return moved


def rotate_polyhedron(
    polyhedron: ConvexPolyhedron,
    axis: np.ndarray = None,
    angle_deg: float = 90.0,
    center: Optional[np.ndarray] = None,
) -> ConvexPolyhedron:
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        raise ValueError("Axis vector cannot be zero")
    axis = axis / axis_norm

    angle_rad = np.radians(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    if center is None:
        center = np.zeros(3)
    else:
        center = np.asarray(center, dtype=float)

    def rotate_vector(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        return (
            v * cos_theta
            + np.cross(axis, v) * sin_theta
            + axis * np.dot(axis, v) * (1.0 - cos_theta)
        )

    rotated_pts = [rotate_vector(pt - center) + center for pt in polyhedron.plane_points]
    rotated_n = [rotate_vector(n) for n in polyhedron.plane_normals]
    rotated = ConvexPolyhedron(
        plane_points=rotated_pts,
        plane_normals=rotated_n,
        vertex_triplets=list(polyhedron.vertex_triplets),
        tolerance=polyhedron.tolerance,
    )
    return rotated
