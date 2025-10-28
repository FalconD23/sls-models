"""
hex_layer.py — ООP-модуль генерации шестиугольной «соты» из
повёрнутых кубов + экспорт в **ASCII-STL c полной точностью координат**.

Главные отличия от предыдущей версии
─────────────────────────────────────
1.  Back-lash по умолчанию = 0 (нет зазора между кубами),
    но можно задать через Layer(edge_backlash=…).
2.  В STL координаты пишутся экспоненциальным форматом «%.9e» —
    ≈ 9-10 значащих цифр float32 → никаких усечений ±1e-6.
3.  Метод `Block.write_facets` рассчитывает нормаль и так же пишет «%.9e».
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, TextIO
import numpy as np, math, matplotlib.pyplot as plt           # type: ignore
from mpl_toolkits.mplot3d import Axes3D                      # noqa: F401

# ════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНОЕ
# ════════════════════════════════════════════════════════════════════

def _rot_body_diag_to_z() -> np.ndarray:
    """Матрица, ставящая пространственную диагональ куба на +Z."""
    alpha = -math.pi / 4
    Rz = np.array([[ math.cos(alpha),-math.sin(alpha),0],
                   [ math.sin(alpha), math.cos(alpha),0],
                   [0,0,1]])
    theta = -math.atan(math.sqrt(2))
    Ry = np.array([[0,1,0],
                   [ math.cos(theta),0, math.sin(theta)],
                   [-math.sin(theta),0, math.cos(theta)]])
    # return Ry @ Rz
    return np.array([[1,0,0],
                    [0,1,0],
                    [0,0,1]])

# !modify: grid's unit - squares 
def _grid_centres(L: int, W: int, s: float) -> List[Tuple[float,float]]:
    """Flat-top hex grid (0,0) — левый-нижний."""
    coords = []
    orients = []
    for row in range(W):
        for col in range(L):
            x = s * col
            y = s * row
            coords.append((x,y))
            orients.append((row+col) % 2)
    return coords, orients

# ════════════════════════════════════════════════════════════════════
#  BLOCK
# ════════════════════════════════════════════════════════════════════

@dataclass
class Block:
    centre:   np.ndarray            # (x,y,z)
    unit_edge:     float                 # длина ребра исходного куба
    backlash: float = 0.0           # относительная «усадка» (< 1 %)
    orientation: int = 0

    _R: np.ndarray = field(init=False, repr=False, default_factory=_rot_body_diag_to_z)
    _base: np.ndarray = field(init=False, repr=False)

    _TRI = [                        # 12 треугольников = 6 граней
        (0,1,2),(0,1,3), 
        (0,2,3),(1,2,3),
    ]

    def __post_init__(self):
        h = self.unit_edge*0.5
        if self.orientation == 0:
            raw = np.array([[-2*h,0,h],[ 2*h,0,h],
                            [0,-2*h,-h],[ 0,2*h,-h]])
        elif self.orientation == 1:
            raw = np.array([[0,-2*h,h],[ 0,2*h,h],
                            [-2*h,0,-h],[ 2*h,0,-h]])
        else:
            raise ValueError(f"Invalid orientation: {self.orientation}")
        self._base = raw @ self._R.T

    # ---------------- public ----------------

    @property
    def vertices(self) -> np.ndarray:
        return self._base + self.centre

    def facets(self):
        V = self.vertices
        return [(V[i],V[j],V[k]) for i,j,k in self._TRI]

    def write_facets(self, fh: TextIO) -> None:
        """Пишет свои 12 треугольников в открытый файловый fh."""
        for v1,v2,v3 in self.facets():
            n = np.cross(v2-v1, v3-v1)
            n /= np.linalg.norm(n) or 1.0
            fh.write(f"  facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}\n")
            fh.write("    outer loop\n")
            for v in (v1,v2,v3):
                fh.write(f"      vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
            fh.write("    endloop\n  endfacet\n")

# ════════════════════════════════════════════════════════════════════
#  LAYER
# ════════════════════════════════════════════════════════════════════

@dataclass
class Layer:
    L: int
    W: int
    grid_side: float
    cz: float = 0.0
    edge_backlash: float = 0.0      # 0 → кубы вплотную

    blocks: List[Block] = field(init=False)

    def __post_init__(self):
        a = self.grid_side           # ребро исходного куба
        coords, orients = _grid_centres(self.L,self.W,self.grid_side)
        self.blocks = []
        for idx, (x,y) in enumerate(coords):
            self.blocks.append(
                Block(np.array([x,y,self.cz]), a, self.edge_backlash, orients[idx])
            )

    def save_ascii_stl(self, fname: str, solid_name="hex_grid_cubes"):
        with open(fname,'w') as fh:
            fh.write(f"solid {solid_name}\n")
            for b in self.blocks: b.write_facets(fh)
            fh.write(f"endsolid {solid_name}\n")
        print(f"[Layer] ASCII-STL записан: {fname}  (blocks={len(self.blocks)})")

# ════════════════════════════════════════════════════════════════════
#  CLI-ЗАПУСК
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse, textwrap
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        Без --stl → интерактивный предпросмотр.
        С  --stl F → сохранение ASCII-STL.
        """))
    p.add_argument("--L", type=int, default=12)
    p.add_argument("--W", type=int, default=12)
    p.add_argument("--hex_side", type=float, default=100.0)
    p.add_argument("--cz", type=float, default=0.0)
    p.add_argument("--backlash", type=float, default=0.0,
                   help="относительная усадка ребра, 0.001 = -0.1 %")
    p.add_argument("--stl", metavar="FILE")
    a = p.parse_args()

    layer = Layer(a.L, a.W, a.hex_side, a.cz, edge_backlash=a.backlash)
    if a.stl:
        layer.save_ascii_stl(a.stl)
    else:
        layer.plot()
