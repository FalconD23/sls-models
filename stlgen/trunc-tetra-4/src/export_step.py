"""STEP (AP214) export for ConvexPolyhedron blocks — по аналогии с export_stl.

Каждый блок -> MANIFOLD_SOLID_BREP из плоских треугольных ADVANCED_FACE;
вершины и рёбра внутри блока переиспользуются (правильная топология BRep).
Все блоки кладутся в одно ADVANCED_BREP_SHAPE_REPRESENTATION.

Файл открывается FreeCAD / любым STEP-просмотрщиком. Единицы — миллиметры.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_STLGEN = Path(__file__).resolve().parent.parent.parent
if str(_STLGEN) not in sys.path:
    sys.path.insert(0, str(_STLGEN))

from geometry.convex_polyhedron import ConvexPolyhedron


def _fmt(v: float) -> str:
    s = f"{v:.6f}"
    return "0." if s == "-0.000000" else s


def _fmt_xyz(x: float, y: float, z: float) -> str:
    return f"({_fmt(x)},{_fmt(y)},{_fmt(z)})"


class _StepBuilder:
    """Нумерует и сериализует сущности ISO-10303-21 (AP214)."""

    def __init__(self) -> None:
        self._lines: List[str] = []
        self._next = 1

    def add(self, text: str) -> int:
        eid = self._next
        self._next += 1
        self._lines.append(f"#{eid}={text};")
        return eid

    def text(self, filename: str) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        header = (
            "ISO-10303-21;\n"
            "HEADER;\n"
            "FILE_DESCRIPTION(('trunc-tetra-4 structure'),'2;1');\n"
            f"FILE_NAME('{filename}','{now}',(''),(''),'stlgen trunc-tetra-4','','');\n"
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 3 1 1 1 }'));\n"
            "ENDSEC;\n"
            "DATA;\n"
        )
        return header + "\n".join(self._lines) + "\nENDSEC;\nEND-ISO-10303-21;\n"


def _block_entities(b: _StepBuilder, poly: ConvexPolyhedron) -> int:
    """Один блок -> MANIFOLD_SOLID_BREP. Возвращает id сущности."""
    verts = np.asarray(poly.vertices, dtype=float)
    faces = np.asarray(poly.get_faces(), dtype=float)  # (n_faces, 3, 3)
    faces = faces.copy()

    # scipy hull не гарантирует согласованный обход симплексов: разворачиваем
    # треугольники так, чтобы нормаль смотрела наружу (блоки выпуклые).
    center = verts.mean(axis=0)
    for t in range(faces.shape[0]):
        tri = faces[t]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if float(np.dot(n, tri.mean(axis=0) - center)) < 0.0:
            faces[t] = tri[::-1]

    # уникальные вершины блока (по координатам с округлением)
    key2idx: Dict[Tuple[float, float, float], int] = {}
    vertex_ids: List[int] = []
    row_idx: List[int] = []  # номер уникальной вершины для каждой строки verts
    for row in verts:
        key = (round(row[0], 6), round(row[1], 6), round(row[2], 6))
        if key not in key2idx:
            i = len(vertex_ids)
            key2idx[key] = i
            pid = b.add(f"CARTESIAN_POINT('',{_fmt_xyz(*row)})")
            vertex_ids.append(b.add(f"VERTEX_POINT('',#{pid})"))
        row_idx.append(key2idx[key])

    # уникальные рёбра: key = (i, j) отсортированная пара вершин
    edge_ids: Dict[Tuple[int, int], Tuple[int, int, int]] = {}  # key -> (edge_id, a, b)

    # координаты уникальных вершин
    uniq_xyz: List[Tuple[float, float, float]] = []
    seen: Dict[Tuple[float, float, float], Tuple[float, float, float]] = {}
    for row in verts:
        k = (round(row[0], 6), round(row[1], 6), round(row[2], 6))
        if k not in seen:
            seen[k] = (float(row[0]), float(row[1]), float(row[2]))
    uniq_xyz = list(seen.values())
    # порядок uniq_xyz совпадает с порядком создания vertex_ids

    def get_edge(i: int, j: int) -> Tuple[int, bool]:
        """Возвращает (id EDGE_CURVE, совпадает ли направление обхода с геометрией ребра)."""
        key = (i, j) if i < j else (j, i)
        entry = edge_ids.get(key)
        if entry is None:
            a = np.array(uniq_xyz[i], dtype=float)
            c = np.array(uniq_xyz[j], dtype=float)
            d = c - a
            mag = float(np.linalg.norm(d))
            dvec = d / mag
            pt = b.add(f"CARTESIAN_POINT('',{_fmt_xyz(*a)})")
            dr = b.add(f"DIRECTION('',{_fmt_xyz(*dvec)})")
            vec = b.add(f"VECTOR('',#{dr},{_fmt(mag)})")
            line = b.add(f"LINE('',#{pt},#{vec})")
            eid = b.add(f"EDGE_CURVE('',#{vertex_ids[i]},#{vertex_ids[j]},#{line},.T.)")
            entry = (eid, i, j)
            edge_ids[key] = entry
        return entry[0], (i == entry[1] and j == entry[2])

    face_ids: List[int] = []
    for tri in faces:
        # индексы уникальных вершин треугольника
        local = []
        for row in tri:
            k = (round(row[0], 6), round(row[1], 6), round(row[2], 6))
            local.append(key2idx[k])
        centroid = tri.mean(axis=0)
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        n = n / np.linalg.norm(n)
        oe_ids = []
        for t in range(3):
            i, j = local[t], local[(t + 1) % 3]
            eid, forward = get_edge(i, j)
            oe_ids.append(b.add(f"ORIENTED_EDGE('',*,*,#{eid},{'.T.' if forward else '.F.'})"))
        loop = b.add(f"EDGE_LOOP('',({','.join(f'#{o}' for o in oe_ids)}))")
        bound = b.add(f"FACE_OUTER_BOUND('',#{loop},.T.)")
        cpt = b.add(f"CARTESIAN_POINT('',{_fmt_xyz(*centroid)})")
        cdir = b.add(f"DIRECTION('',{_fmt_xyz(*n)})")
        axis = b.add(f"AXIS2_PLACEMENT_3D('',#{cpt},#{cdir},$)")
        plane = b.add(f"PLANE('',#{axis})")
        face_ids.append(b.add(f"ADVANCED_FACE('',(#{bound}),#{plane},.T.)"))

    shell = b.add(f"CLOSED_SHELL('',({','.join(f'#{f}' for f in face_ids)}))")
    return b.add(f"MANIFOLD_SOLID_BREP('',#{shell})")


def export_step(blocks: Sequence[ConvexPolyhedron], filename: str | Path) -> Path:
    """Экспортирует список блоков в STEP AP214 (мм). Возвращает путь к файлу."""
    b = _StepBuilder()
    app_ctx = b.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    b.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app_ctx})")
    prod_ctx = b.add(f"PRODUCT_CONTEXT('',#{app_ctx},'mechanical')")
    prod = b.add("PRODUCT('trunc-tetra-4','trunc-tetra-4 plate','',(" + f"#{prod_ctx}" + "))")
    pdf = b.add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
    pdc = b.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app_ctx},'design')")
    pd = b.add(f"PRODUCT_DEFINITION('design','',#{pdf},#{pdc})")

    length = b.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    solid_angle = b.add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    plane_angle = b.add("(NAMED_UNIT(*)SI_UNIT($,.RADIAN.)PLANE_ANGLE_UNIT())")
    unc = b.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-05),#{length},'distance_accuracy_value','')")
    ctx = b.add(
        f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
        f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{solid_angle},#{plane_angle}))"
        f"REPRESENTATION_CONTEXT('Context #1','3D Context'))"
    )

    solids = [_block_entities(b, poly) for poly in blocks]
    origin = b.add("CARTESIAN_POINT('',(0.,0.,0.))")
    zdir = b.add("DIRECTION('',(0.,0.,1.))")
    xdir = b.add("DIRECTION('',(1.,0.,0.))")
    place = b.add(f"AXIS2_PLACEMENT_3D('',#{origin},#{zdir},#{xdir})")
    rep = b.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',({','.join(f'#{s}' for s in solids)}),#{ctx})")
    pds = b.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd})")
    b.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(b.text(path.name), encoding="ascii")
    return path


if __name__ == "__main__":
    # самопроверка: один блок DOUBLE
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from blocks import BlockKind, create_block
    from config import Tetra4Config

    out = Path(__file__).resolve().parent.parent / "out" / "step_selftest.step"
    poly = create_block(BlockKind.DOUBLE_2H, Tetra4Config(H=26.75, L=148.0))
    export_step([poly], out)
    print(f"OK: {out}")
