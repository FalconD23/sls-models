"""trunc-tetra-4 sandbox sources. Copy into stlgen later (see 02_workflow.md)."""

try:
    from .config import Tetra4Config, BlockKind
    from .blocks import (
        create_truncated_tetrahedron,
        create_block,
        translate_polyhedron,
        rotate_polyhedron,
    )
    from .placement import Placement, generate_placements
    from .layer_gen import Tetra4LayerGen
except ImportError:
    from config import Tetra4Config, BlockKind
    from blocks import (
        create_truncated_tetrahedron,
        create_block,
        translate_polyhedron,
        rotate_polyhedron,
    )
    from placement import Placement, generate_placements
    from layer_gen import Tetra4LayerGen

__all__ = [
    "Tetra4Config",
    "BlockKind",
    "create_truncated_tetrahedron",
    "create_block",
    "translate_polyhedron",
    "rotate_polyhedron",
    "Placement",
    "generate_placements",
    "Tetra4LayerGen",
]
