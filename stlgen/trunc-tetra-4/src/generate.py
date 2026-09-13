"""Facade: layer size + block size → STL or (later) STEP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
_STLGEN = _SRC.parent.parent
if str(_STLGEN) not in sys.path:
    sys.path.insert(0, str(_STLGEN))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Tetra4Config
from export_step import export_step
from export_stl import export_blocks_stl
from layer_gen import Tetra4LayerGen
from visualize import visualize_layer


def generate(
    nx: int,
    ny: int,
    nz: int,
    H: float,
    L: float,
    alpha_deg: float = 25.0,
    output: str | None = None,
    fmt: str = "stl",
    viz: bool = False,
    viz_path: str | None = None,
) -> None:
    config = Tetra4Config(H=H, L=L, alpha_deg=alpha_deg)
    blocks, placements = Tetra4LayerGen(nx, ny, nz, config)
    print(f"Generated {len(blocks)} blocks ({nx}x{ny}x{nz}, H={H}, L={L}, α={alpha_deg})")

    if viz or viz_path:
        visualize_layer(
            blocks,
            placements,
            show=viz,
            save_path=Path(viz_path) if viz_path else None,
            title=f"trunc-tetra-4 {nx}x{ny}x{nz}",
        )

    if output is None:
        return

    if fmt == "stl":
        export_blocks_stl(blocks, output, solid_name="trunc_tetra_4", format="ascii")
    elif fmt == "step":
        export_step(blocks, output)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate trunc-tetra-4 layer")
    p.add_argument("--nx", type=int, default=3, help="двойных блоков по X")
    p.add_argument("--ny", type=int, default=3, help="двойных блоков по Y")
    p.add_argument("--nz", type=int, default=6, help="чётное число слоёв H")
    p.add_argument("--H", type=float, default=10.0)
    p.add_argument("--L", type=float, default=20.0)
    p.add_argument("--alpha", type=float, default=25.0)
    p.add_argument("--stl", dest="stl", default=None, help="output STL path")
    p.add_argument("--format", dest="fmt", choices=("stl", "step"), default="stl")
    p.add_argument("--viz", action="store_true")
    p.add_argument("--viz-path", default=None)
    a = p.parse_args()

    out = a.stl
    if out is None and a.fmt == "stl":
        out = None
    if a.fmt == "step" and out is None:
        out = "out/trunc_tetra_4.step"

    generate(
        nx=a.nx,
        ny=a.ny,
        nz=a.nz,
        H=a.H,
        L=a.L,
        alpha_deg=a.alpha,
        output=out,
        fmt=a.fmt,
        viz=a.viz,
        viz_path=a.viz_path,
    )


if __name__ == "__main__":
    main()
