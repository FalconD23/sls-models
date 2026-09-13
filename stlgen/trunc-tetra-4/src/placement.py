"""Checkerboard placement: doubles on black, large/small on white, even-row shift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from .config import BlockKind, Tetra4Config
except ImportError:
    from config import BlockKind, Tetra4Config


@dataclass
class Placement:
    kind: BlockKind
    i: int
    j: int
    center: np.ndarray
    half_sign: int = 1


def _is_black(i: int, j: int) -> bool:
    return (i + j) % 2 == 0


def generate_placements(
    nx_bl: int,
    ny_bl: int,
    nz: int,
    config: Tetra4Config,
) -> List[Placement]:
    """
    nx_bl, ny_bl — число ДВОЙНЫХ блоков по X и Y (не клеток).
    Клеток шахматки: (2*nx_bl − 1) × (2*ny_bl − 1), углы — чёрные.
    nz — чётное число слоёв высоты H.
    Пластина по Z: [z_min, z_max], на обоих торцах — плоские подошвы.

    Правила (по описанию):
    - Чёрные клетки: DOUBLE_2H, одна ориентация. Нечётный ряд j — центры на
      z = z_min + H + 2H·m; чётный ряд — сдвиг +H.
    - Двойные проникают через соседние слои; на торцах, где не помещается
      целый двойной, ставится HALF_H (замыкание чётности).
    - Белые клетки: LARGE_H и SMALL_H чередуются по слою k и отвечают
      боковым граням чёрных блоков.
    """
    if nx_bl < 1 or ny_bl < 1:
        raise ValueError("nx_bl, ny_bl must be >= 1")
    if nz < 2 or nz % 2 != 0:
        raise ValueError(f"nz must be even (2 per two-layer period), got {nz}")

    H = config.H
    pitch = config.cell_pitch
    tol = config.tolerance
    nz_layers = nz // 2

    n_cells_x = 2 * nx_bl - 1
    n_cells_y = 2 * ny_bl - 1
    z_min = -H * nz_layers
    z_max = H * nz_layers

    placements: List[Placement] = []

    for i in range(n_cells_x):
        for j in range(n_cells_y):
            x = (i - (n_cells_x - 1) / 2.0) * pitch
            y = (j - (n_cells_y - 1) / 2.0) * pitch
            even_row = (j % 2 == 0)

            if _is_black(i, j):
                # --- DOUBLE_2H ---
                # Нечётный ряд: центры z_min+H, z_min+3H, ... — заполняют пластину целиком.
                # Чётный ряд: сдвиг +H → двойные внутри, HALF снизу и сверху.
                m = 0
                while True:
                    zc = z_min + H + 2.0 * H * m + (H if even_row else 0.0)
                    z_lo, z_hi = zc - H, zc + H
                    if z_lo > z_max + tol:
                        break
                    if z_lo >= z_min - tol and z_hi <= z_max + tol:
                        placements.append(
                            Placement(
                                kind=BlockKind.DOUBLE_2H,
                                i=i,
                                j=j,
                                center=np.array([x, y, zc]),
                            )
                        )
                    m += 1
                    if m > nz + 2:
                        break

                # HALF только там, где двойной не помещается.
                # Чётный ряд: щель H снизу и сверху. Нечётный: щелей нет (nz чётный).
                if even_row:
                    placements.append(
                        Placement(
                            kind=BlockKind.HALF_H,
                            i=i,
                            j=j,
                            center=np.array([x, y, z_min]),
                            half_sign=1,  # верхняя половина блока под пластиной
                        )
                    )
                    placements.append(
                        Placement(
                            kind=BlockKind.HALF_H,
                            i=i,
                            j=j,
                            center=np.array([x, y, z_max]),
                            half_sign=-1,  # нижняя половина блока над пластиной
                        )
                    )
            else:
                # Белые клетки НЕ сдвигаются с рядом: они отвечают боковым граням
                # и чётных, и нечётных чёрных соседей сразу. Каждый слой k — один блок H.
                for k in range(nz):
                    zc = z_min + (k + 0.5) * H
                    kind = BlockKind.LARGE_H if (k % 2 == 0) else BlockKind.SMALL_H
                    placements.append(
                        Placement(kind=kind, i=i, j=j, center=np.array([x, y, zc]))
                    )

    return placements


def cell_xy(i: int, j: int, nx_bl: int, ny_bl: int, pitch: float) -> tuple[float, float]:
    n_cells_x = 2 * nx_bl - 1
    n_cells_y = 2 * ny_bl - 1
    x = (i - (n_cells_x - 1) / 2.0) * pitch
    y = (j - (n_cells_y - 1) / 2.0) * pitch
    return x, y


def black_stack_for_cell(
    i: int,
    j: int,
    nz: int,
    config: Tetra4Config,
    nx_bl: int,
    ny_bl: int,
) -> List[Placement]:
    """DOUBLE/HALF колонка одной чёрной клетки (в т.ч. виртуальная за краем пластины)."""
    if (i + j) % 2 != 0:
        raise ValueError(f"({i},{j}) is not black")
    H = config.H
    nz_layers = nz // 2
    z_min = -H * nz_layers
    z_max = H * nz_layers
    tol = config.tolerance
    x, y = cell_xy(i, j, nx_bl, ny_bl, config.cell_pitch)
    even_row = j % 2 == 0
    out: List[Placement] = []
    m = 0
    while True:
        zc = z_min + H + 2.0 * H * m + (H if even_row else 0.0)
        z_lo, z_hi = zc - H, zc + H
        if z_lo > z_max + tol:
            break
        if z_lo >= z_min - tol and z_hi <= z_max + tol:
            out.append(Placement(kind=BlockKind.DOUBLE_2H, i=i, j=j, center=np.array([x, y, zc])))
        m += 1
        if m > nz + 2:
            break
    if even_row:
        out.append(
            Placement(
                kind=BlockKind.HALF_H,
                i=i,
                j=j,
                center=np.array([x, y, z_min]),
                half_sign=1,
            )
        )
        out.append(
            Placement(
                kind=BlockKind.HALF_H,
                i=i,
                j=j,
                center=np.array([x, y, z_max]),
                half_sign=-1,
            )
        )
    return out
