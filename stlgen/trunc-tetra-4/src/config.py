"""Geometry configuration for trunc-tetra-4 blocks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class BlockKind(str, Enum):
    DOUBLE_2H = "DOUBLE_2H"
    HALF_H = "HALF_H"
    LARGE_H = "LARGE_H"
    SMALL_H = "SMALL_H"


@dataclass(frozen=True)
class Tetra4Config:
    """
    H — высота слоя (= полвысоты двойного блока).
    L — шаг клетки шахматки.
    alpha_deg — угол, задающий разницу большого и маленького блоков.
    """

    H: float
    L: float
    alpha_deg: float = 25.0
    tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if self.H <= 0:
            raise ValueError(f"H must be positive, got {self.H}")
        if self.L <= 0:
            raise ValueError(f"L must be positive, got {self.L}")
        if self.alpha_deg < 0 or self.alpha_deg >= 90:
            raise ValueError(f"alpha_deg must be in [0, 90), got {self.alpha_deg}")

    @property
    def alpha_rad(self) -> float:
        return float(np.radians(self.alpha_deg))

    @property
    def tan_alpha(self) -> float:
        return float(np.tan(self.alpha_rad))

    @property
    def a_double(self) -> float:
        """Ребро исходного тетраэдра для двойного блока.

        Пояс z=0 — квадрат со стороной a/2; хотим сторону ≈ L.
        Также a должно быть достаточно большим, чтобы срез ±H попал внутрь тетраэдра.
        """
        a_from_cell = 2.0 * self.L
        a_from_cut = 2.0 * self.H * np.sqrt(2.0) * 1.1
        return float(max(a_from_cell, a_from_cut))

    @property
    def cell_pitch(self) -> float:
        """Фактический шаг шахматки по XY.

        Для высокого узкого блока ``L`` может быть меньше минимального
        центрального сечения тетраэдра. В таком случае увеличиваем только
        шаг размещения, иначе четыре чёрных соседа полностью съедают белый
        блок и пересечение полупространств получается пустым.
        """
        return float(self.a_double / 2.0)

    @property
    def a_large(self) -> float:
        return float(self.a_double + 2.0 * self.H * self.tan_alpha)

    @property
    def a_small(self) -> float:
        return float(max(self.a_double * 0.25, self.a_double - 2.0 * self.H * self.tan_alpha))
