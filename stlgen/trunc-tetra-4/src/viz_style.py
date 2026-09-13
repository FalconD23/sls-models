"""Настройки matplotlib-отображения слоя (цвета, контуры, плоскость разреза)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

try:
    from .config import BlockKind
except ImportError:
    from config import BlockKind


@dataclass
class VizStyle:
    """Крутить здесь, если плохо видно контуры / разрез."""

    figsize: tuple = (16, 12)
    dpi: int = 170

    face_alpha: float = 0.5
    edge_color: str = "#111111"
    edge_width: float = 0.1
    edge_alpha: float = 0.5

    kind_colors: Dict[BlockKind, str] = field(
        default_factory=lambda: {
            BlockKind.DOUBLE_2H: "steelblue",
            BlockKind.HALF_H: "gold",
            BlockKind.LARGE_H: "mediumseagreen",
            BlockKind.SMALL_H: "tomato",
        }
    )
    fallback_color: str = "cyan"
    legend_alpha: float = 0.55

    cut_face_color: str = "#9aa3ad"
    cut_face_alpha: float = 0.1
    cut_edge_color: str = "#4a5560"
    cut_edge_width: float = 0.8
    cut_edge_alpha: float = 0.35

    # Только отрисовка: линейный масштаб каждого блока от его центроида.
    # 0.99 = −1%, чтобы грани не z-fight'или в matplotlib. STL не трогает.
    display_scale: float = 0.99


# дефолт; можно подменить: visualize_layer(..., style=VizStyle(edge_width=2.0))
VIZ = VizStyle()
