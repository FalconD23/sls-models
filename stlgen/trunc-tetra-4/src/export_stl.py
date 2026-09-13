"""STL export wrapper around stlgen.STLExporter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

_STLGEN = Path(__file__).resolve().parent.parent.parent
if str(_STLGEN) not in sys.path:
    sys.path.insert(0, str(_STLGEN))

from export.stl_exporter import STLExporter
from geometry.convex_polyhedron import ConvexPolyhedron


def export_blocks_stl(
    blocks: Sequence[ConvexPolyhedron],
    filename: str,
    solid_name: str = "trunc_tetra_4",
    format: str = "ascii",
) -> None:
    all_blocks = []
    for poly in blocks:
        faces = poly.get_faces()
        if faces:
            all_blocks.append(faces)
    if not all_blocks:
        raise ValueError("No faces to export")

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    STLExporter().write_stl(
        blocks=all_blocks,
        filename=filename,
        solid_name=solid_name,
        format=format,
    )
