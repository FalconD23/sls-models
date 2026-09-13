"""Matplotlib visualization for one polyhedron or a whole layer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

_STLGEN = Path(__file__).resolve().parent.parent.parent
if str(_STLGEN) not in sys.path:
    sys.path.insert(0, str(_STLGEN))

from geometry.convex_polyhedron import ConvexPolyhedron

try:
    from .config import BlockKind
    from .placement import Placement
    from .viz_style import VIZ, VizStyle
except ImportError:
    from config import BlockKind
    from placement import Placement
    from viz_style import VIZ, VizStyle


def _display_center(poly: ConvexPolyhedron, placement: Optional[Placement]) -> np.ndarray:
    if poly.vertices is not None and len(poly.vertices) > 0:
        return np.mean(poly.vertices, axis=0)
    if placement is not None:
        return np.asarray(placement.center, dtype=float)
    return np.zeros(3)


def _scale_about(points, center: np.ndarray, scale: float):
    pts = np.asarray(points, dtype=float)
    if scale == 1.0:
        return pts
    c = np.asarray(center, dtype=float)
    return c + scale * (pts - c)


def visualize_layer(
    blocks: Sequence[ConvexPolyhedron],
    placements: Optional[Sequence[Placement]] = None,
    ax=None,
    show: bool = False,
    save_path: Optional[Path] = None,
    title: str = "trunc-tetra-4",
    cut_plane: Optional[tuple] = None,
    style: Optional[VizStyle] = None,
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull

    st = style or VIZ

    if ax is None:
        fig = plt.figure(figsize=st.figsize)
        ax = fig.add_subplot(111, projection="3d")

    all_vertices: List[np.ndarray] = []

    for idx, poly in enumerate(blocks):
        pl = placements[idx] if placements is not None else None
        kind = pl.kind if pl is not None else None
        color = st.kind_colors.get(kind, st.fallback_color) if kind is not None else st.fallback_color
        center = _display_center(poly, pl)
        scale = st.display_scale

        if poly.faces:
            faces = [
                _scale_about(face, center, scale) for face in poly.faces
            ]
            ax.add_collection3d(
                Poly3DCollection(
                    faces,
                    alpha=st.face_alpha,
                    facecolor=color,
                    edgecolor=st.edge_color,
                    linewidths=st.edge_width,
                    linestyles="solid",
                )
            )

        if poly.vertices is not None and len(poly.vertices) >= 4:
            verts = _scale_about(poly.vertices, center, scale)
            all_vertices.append(np.asarray(poly.vertices, dtype=float))
            try:
                hull = ConvexHull(verts)
                edges = set()
                for simplex in hull.simplices:
                    edges.add(tuple(sorted((simplex[0], simplex[1]))))
                    edges.add(tuple(sorted((simplex[1], simplex[2]))))
                    edges.add(tuple(sorted((simplex[2], simplex[0]))))
                for e in edges:
                    v1, v2 = verts[e[0]], verts[e[1]]
                    ax.plot(
                        [v1[0], v2[0]],
                        [v1[1], v2[1]],
                        [v1[2], v2[2]],
                        color=st.edge_color,
                        linewidth=st.edge_width,
                        alpha=st.edge_alpha,
                        solid_capstyle="round",
                    )
            except Exception:
                pass

    if all_vertices:
        pts = np.vstack(all_vertices)
        c = np.mean(pts, axis=0)
        r = np.max(np.linalg.norm(pts - c, axis=1)) * 1.2
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        if cut_plane is not None:
            _draw_cut_plane(ax, cut_plane[0], cut_plane[1], r, st)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    if placements is not None:
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(facecolor=st.kind_colors[k], alpha=st.legend_alpha, label=k.value)
                for k in BlockKind
            ],
            loc="upper right",
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=st.dpi, bbox_inches="tight")

    if show:
        plt.show()

    return ax


def signed_distance_to_plane(point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> float:
    n = np.asarray(plane_normal, dtype=float)
    nlen = np.linalg.norm(n)
    if nlen < 1e-15:
        raise ValueError("plane_normal cannot be zero")
    n = n / nlen
    return float(np.dot(n, np.asarray(point, dtype=float) - np.asarray(plane_point, dtype=float)))


def filter_blocks_by_plane(
    blocks: Sequence[ConvexPolyhedron],
    placements: Sequence[Placement],
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    keep_negative: bool = True,
    eps: float = 1e-9,
) -> tuple[list, list]:
    """Оставить блоки, чей центр по одну сторону от плоскости n·(x−p)=0.

    keep_negative=True → оставляем n·(c−p) <= 0 (половина «под» нормалью).
    """
    kept_b: list = []
    kept_p: list = []
    for poly, pl in zip(blocks, placements):
        d = signed_distance_to_plane(pl.center, plane_point, plane_normal)
        if keep_negative:
            ok = d <= eps
        else:
            ok = d >= -eps
        if ok:
            kept_b.append(poly)
            kept_p.append(pl)
    return kept_b, kept_p


def _draw_cut_plane(ax, plane_point, plane_normal, span: float, style: VizStyle) -> None:
    """Полупрозрачный четырёхугольник режущей плоскости."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    n = np.asarray(plane_normal, dtype=float)
    n = n / np.linalg.norm(n)
    p0 = np.asarray(plane_point, dtype=float)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    s = span
    corners = [
        p0 + s * u + s * v,
        p0 + s * u - s * v,
        p0 - s * u - s * v,
        p0 - s * u + s * v,
    ]
    coll = Poly3DCollection(
        [corners],
        alpha=style.cut_face_alpha,
        facecolor=style.cut_face_color,
        edgecolor=style.cut_edge_color,
        linewidths=style.cut_edge_width,
    )
    coll.set_alpha(style.cut_face_alpha)
    ax.add_collection3d(coll)
