"""Assemble ConvexPolyhedron blocks from placements."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

_STLGEN = Path(__file__).resolve().parent.parent.parent
if str(_STLGEN) not in sys.path:
    sys.path.insert(0, str(_STLGEN))

from geometry.convex_polyhedron import ConvexPolyhedron

try:
    from .blocks import create_block, create_white_from_neighbors
    from .config import BlockKind, Tetra4Config
    from .placement import (
        Placement,
        black_stack_for_cell,
        generate_placements,
    )
except ImportError:
    from blocks import create_block, create_white_from_neighbors
    from config import BlockKind, Tetra4Config
    from placement import (
        Placement,
        black_stack_for_cell,
        generate_placements,
    )


def _z_span(p: Placement, H: float) -> tuple[float, float]:
    zc = float(p.center[2])
    if p.kind == BlockKind.DOUBLE_2H:
        return zc - H, zc + H
    if p.kind == BlockKind.HALF_H:
        if p.half_sign >= 0:
            return zc, zc + H
        return zc - H, zc
    return zc - 0.5 * H, zc + 0.5 * H


def _black_poly_at(
    i: int,
    j: int,
    z_mid: float,
    nx_bl: int,
    ny_bl: int,
    nz: int,
    config: Tetra4Config,
    cache: dict,
) -> ConvexPolyhedron:
    key = (i, j)
    if key not in cache:
        stack = black_stack_for_cell(i, j, nz, config, nx_bl, ny_bl)
        cache[key] = [
            (p, create_block(p.kind, config, center=p.center, half_sign=p.half_sign))
            for p in stack
        ]
    H = config.H
    eps = config.tolerance * 10
    for p, poly in cache[key]:
        lo, hi = _z_span(p, H)
        if lo - eps <= z_mid <= hi + eps:
            return poly
    raise RuntimeError(f"Нет чёрного блока в ({i},{j}) на z={z_mid}")


def Tetra4LayerGen(
    nx_bl: int,
    ny_bl: int,
    nz: int,
    config: Tetra4Config,
) -> Tuple[List[ConvexPolyhedron], List[Placement]]:
    """nx_bl, ny_bl — число двойных блоков по осям; nz — чётное число слоёв H."""
    placements = generate_placements(nx_bl, ny_bl, nz, config)
    black_cache: dict = {}
    blocks: List[ConvexPolyhedron] = []
    H = config.H

    for p in placements:
        if p.kind in (BlockKind.DOUBLE_2H, BlockKind.HALF_H):
            blocks.append(
                create_block(
                    kind=p.kind,
                    config=config,
                    center=p.center,
                    half_sign=p.half_sign,
                )
            )
            continue

        z_mid = float(p.center[2])
        z_lo, z_hi = z_mid - 0.5 * H, z_mid + 0.5 * H
        neighbors = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbors.append(
                _black_poly_at(
                    p.i + di,
                    p.j + dj,
                    z_mid,
                    nx_bl,
                    ny_bl,
                    nz,
                    config,
                    black_cache,
                )
            )
        blocks.append(
            create_white_from_neighbors(
                center=p.center,
                z_lo=z_lo,
                z_hi=z_hi,
                neighbor_polys=neighbors,
                tolerance=config.tolerance,
            )
        )
    return blocks, placements
