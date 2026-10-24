"""Пластина trunc-tetra-4 под габарит 1370x1370x107 мм.

Толщина задаётся точно: nz * H = thickness.
Шаг L подбирается так, чтобы итоговый bbox по XY совпал с целевым —
блоки выступают за сетку клеток, поэтому n_cells * L меньше габарита.

STL и PNG по умолчанию пишутся в stlgen/collection.

Запуск:
    python trunc-tetra-4/scripts/gen_plate_1370.py                 # STL + PNG
    python trunc-tetra-4/scripts/gen_plate_1370.py --format step   # STEP
    python trunc-tetra-4/scripts/gen_plate_1370.py --show
    python trunc-tetra-4/scripts/gen_plate_1370.py --nx 6 --ny 6   # ~400 блоков
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
_STLGEN = _SRC.parent.parent
for p in (str(_STLGEN), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import Tetra4Config
from generate import generate
from layer_gen import Tetra4LayerGen

PLATE_XY = 1370.0
PLATE_Z = 107.0


def layer_bbox(nx: int, ny: int, nz: int, H: float, L: float, alpha_deg: float) -> np.ndarray:
    config = Tetra4Config(H=H, L=L, alpha_deg=alpha_deg)
    blocks, _ = Tetra4LayerGen(nx, ny, nz, config)
    verts = np.vstack([b.vertices for b in blocks])
    return verts.max(axis=0) - verts.min(axis=0)


def fit_pitch(
    nx: int,
    ny: int,
    nz: int,
    H: float,
    alpha_deg: float,
    target_xy: float,
    tol: float = 0.05,
    max_iter: int = 40,
) -> float:
    """Подогнать L так, чтобы габарит по X совпал с target_xy."""
    n_cells = 2 * nx - 1
    L = target_xy / n_cells
    for _ in range(max_iter):
        err = layer_bbox(nx, ny, nz, H, L, alpha_deg)[0] - target_xy
        if abs(err) < tol:
            break
        L -= err / n_cells
    return L


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=5, help="двойных блоков по X")
    p.add_argument("--ny", type=int, default=5, help="двойных блоков по Y")
    p.add_argument("--nz", type=int, default=4, help="чётное число слоёв H")
    p.add_argument("--alpha", type=float, default=25.0)
    p.add_argument("--xy", type=float, default=PLATE_XY, help="габарит пластины по X и Y, мм")
    p.add_argument("--thickness", type=float, default=PLATE_Z, help="толщина пластины, мм")
    p.add_argument(
        "--format",
        dest="fmt",
        choices=("stl", "step"),
        default="stl",
        help="формат экспорта (stl по умолчанию)",
    )
    p.add_argument(
        "--stl",
        default=None,
        help="путь к файлу (по умолчанию stlgen/collection/plate_<габарит>.<fmt>)",
    )
    p.add_argument("--show", action="store_true", help="показать окно matplotlib")
    a = p.parse_args()

    H = a.thickness / a.nz
    L = fit_pitch(a.nx, a.ny, a.nz, H, a.alpha, a.xy)
    config = Tetra4Config(H=H, L=L, alpha_deg=a.alpha)
    ext = layer_bbox(a.nx, a.ny, a.nz, H, L, a.alpha)

    out_dir = _STLGEN / "collection"
    out_dir.mkdir(parents=True, exist_ok=True)
    stl_path = a.stl or str(
        out_dir / f"plate_{a.xy:.0f}x{a.xy:.0f}x{a.thickness:.0f}.{a.fmt}"
    )
    png_path = str(Path(stl_path).with_suffix(".png"))

    print(f"клеток шахматки: {2 * a.nx - 1} x {2 * a.ny - 1}, слоёв H: {a.nz}")
    print(f"H = {H:.4f} мм, L = {L:.4f} мм, фактический шаг XY = {config.cell_pitch:.4f} мм")
    print(f"блок: ширина/высота = {L / (2 * H):.2f} (на фото образца ~2-3)")
    print(f"габарит: {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} мм")

    generate(
        nx=a.nx,
        ny=a.ny,
        nz=a.nz,
        H=H,
        L=L,
        alpha_deg=a.alpha,
        output=stl_path,
        fmt=a.fmt,
        viz=a.show,
        viz_path=png_path,
    )
    print(f"{a.fmt.upper()}: {stl_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
