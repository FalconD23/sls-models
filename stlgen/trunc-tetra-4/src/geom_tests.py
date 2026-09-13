"""
Тесты прототипа trunc-tetra-4.

По умолчанию в main() только test_truncated_tetrahedron — один блок,
matplotlib + STL. Остальное раскомментировать после зелёного шага 0.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from collections import Counter
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
_STLGEN = _ROOT.parent
_OUT = _ROOT / "out"
_OUT.mkdir(parents=True, exist_ok=True)

for p in (str(_STLGEN), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from blocks import create_block, create_truncated_tetrahedron
from config import BlockKind, Tetra4Config
from export_stl import export_blocks_stl
from layer_gen import Tetra4LayerGen
from placement import generate_placements
from viz_style import VIZ
from visualize import filter_blocks_by_plane, visualize_layer


def _try_interactive_backend() -> str:
    """Pick a GUI backend for an interactive window (works from terminal).

    Order: Qt6/Qt5 → Tk. Returns backend name or empty string if none worked.
    Call BEFORE importing matplotlib.pyplot.
    """
    import matplotlib

    for name in ("Qt6Agg", "Qt5Agg", "TkAgg"):
        try:
            matplotlib.use(name, force=True)
            return name
        except Exception:
            continue
    return ""


def test_truncated_tetrahedron(
    show: bool = True,
    interactive_backend: bool = False,
) -> None:
    print("=" * 60)
    print("Тест: Усечённый тетраэдр")
    print("=" * 60)

    a = 40.0
    half_height = 10.0
    print(f"\nПараметры: a={a}, half_height={half_height}")

    poly, params = create_truncated_tetrahedron(a=a, half_height=half_height)
    print(f"\n✓ Создан полиэдр: {poly}")
    print(f"  Плоскостей: {len(poly.plane_points)} (должно быть 6)")
    print(f"  Троек вершин: {len(poly.vertex_triplets)} (должно быть 8)")
    print(f"  h0 несрезанного тетраэдра: {params['h0']:.4f}")

    print("\nВычисление вершин...")
    vertices = poly.build_vertices()
    print(f"✓ Вершины построены: {len(vertices)} (ожидается 8)")
    assert len(vertices) >= 4, f"Need at least 4 vertices, got {len(vertices)}"
    if len(vertices) != 8:
        print(f"   Warning: expected 8 vertices, got {len(vertices)}")

    print("\nПостроение выпуклой оболочки...")
    faces = poly.build_convex_hull()
    print(f"✓ Convex hull: {len(faces)} треугольных граней")
    assert len(faces) > 0, "Should have at least some faces"

    stl_name = f"trunc_tet_a{a}_h{half_height}.stl"
    stl_path = _OUT / stl_name
    print(f"\nЭкспорт в STL: {stl_path.name}")
    poly.to_stl(str(stl_path), solid_name=f"trunc_tet_a{a}_h{half_height}", format="ascii")
    assert stl_path.exists() and stl_path.stat().st_size > 0
    print(f"✓ Размер файла: {stl_path.stat().st_size} байт")

    try:
        if interactive_backend:
            backend = _try_interactive_backend()
            if backend:
                print(f"  Бэкенд: {backend} (интерактивное окно)")

        import matplotlib.pyplot as plt

        print("\nВизуализация...")
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection="3d")
        poly.visualize(
            ax=ax,
            show_planes=True,
            show_edges=True,
            show_vertices=True,
            show_polyhedron=True,
            scale_normal=5.0,
        )
        ax.set_title(f"truncated tetrahedron: a={a}, half_height={half_height}", fontsize=14)
        png_path = _OUT / f"trunc_tet_a{a}_h{half_height}.png"
        plt.savefig(str(png_path), dpi=150, bbox_inches="tight")
        print(f"✓ Визуализация сохранена: {png_path.name}")
        if show:
            print("  Отображение графика... (закройте окно для продолжения)")
            plt.show()
        else:
            plt.close(fig)
    except ImportError:
        print("⚠️ Matplotlib не доступен, визуализация пропущена")
    except Exception as e:
        print(f"⚠️ Ошибка при визуализации: {e}")

    print("\n" + "=" * 60)
    print("✅ test_truncated_tetrahedron пройден")
    print("=" * 60)


def test_four_block_types(
    show: bool = False,
    interactive_backend: bool = False,
) -> None:
    print("=" * 60)
    print("Тест: четыре типа блоков")
    print("=" * 60)

    config = Tetra4Config(H=10.0, L=20.0, alpha_deg=25.0)
    kinds = [
        (BlockKind.DOUBLE_2H, 1),
        (BlockKind.HALF_H, 1),
        (BlockKind.HALF_H, -1),
        (BlockKind.LARGE_H, 1),
        (BlockKind.SMALL_H, 1),
    ]

    for kind, half_sign in kinds:
        poly = create_block(kind, config, half_sign=half_sign)
        tag = f"{kind.value}_s{half_sign}"
        print(f"  {tag}: vertices={len(poly.vertices)}, faces={len(poly.faces)}")
        assert len(poly.vertices) >= 4
        assert len(poly.faces) > 0
        poly.to_stl(str(_OUT / f"{tag}.stl"), solid_name=tag, format="ascii")

        try:
            if interactive_backend:
                backend = _try_interactive_backend()
                if backend:
                    print(f"  Бэкенд: {backend}")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            poly.visualize(ax=ax, show_planes=False, scale_normal=3.0)
            ax.set_title(tag)
            plt.savefig(str(_OUT / f"{tag}.png"), dpi=120, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)
        except Exception as e:
            print(f"   viz skip: {e}")

    print("✅ test_four_block_types пройден")


def test_halves_match_double(H: float = 10.0, L: float = 20.0, alpha_deg: float = 25.0) -> None:
    print("=" * 60)
    print("Тест: две половинки = двойной по Z")
    print("=" * 60)

    config = Tetra4Config(H=H, L=L, alpha_deg=alpha_deg)
    double = create_block(BlockKind.DOUBLE_2H, config)
    upper = create_block(BlockKind.HALF_H, config, half_sign=1)
    lower = create_block(BlockKind.HALF_H, config, half_sign=-1)

    z_d = (double.vertices[:, 2].min(), double.vertices[:, 2].max())
    z_u = (upper.vertices[:, 2].min(), upper.vertices[:, 2].max())
    z_l = (lower.vertices[:, 2].min(), lower.vertices[:, 2].max())
    z_union = (min(z_u[0], z_l[0]), max(z_u[1], z_l[1]))

    print(f"  DOUBLE Z: {z_d}")
    print(f"  HALF+  Z: {z_u}")
    print(f"  HALF-  Z: {z_l}")
    print(f"  union  Z: {z_union}")

    assert np.isclose(z_d[0], z_union[0], atol=1e-4), (z_d, z_union)
    assert np.isclose(z_d[1], z_union[1], atol=1e-4), (z_d, z_union)
    assert z_u[0] >= -1e-4
    assert z_l[1] <= 1e-4
    print("✅ test_halves_match_double пройден")


def test_placement(
    nx_bl: int = 3,
    ny_bl: int = 3,
    nz: int = 6,
    H: float = 10.0,
    L: float = 20.0,
    alpha_deg: float = 25.0,
) -> None:
    print("=" * 60)
    print(f"Тест: placement {nx_bl}×{ny_bl}×{nz}")
    print("=" * 60)

    if nz % 2 != 0:
        raise ValueError(f"nz must be even (2 слоя на период), got {nz}")

    config = Tetra4Config(H=H, L=L, alpha_deg=alpha_deg)
    # nx_bl, ny_bl — двойные; nz — чётное число слоёв H
    placements = generate_placements(nx_bl, ny_bl, nz, config)
    counts = Counter(p.kind for p in placements)
    print(f"  всего: {len(placements)}")
    for k, n in counts.items():
        print(f"    {k.value}: {n}")

    assert counts[BlockKind.DOUBLE_2H] > 0
    assert counts[BlockKind.HALF_H] > 0
    assert counts[BlockKind.LARGE_H] > 0
    assert counts[BlockKind.SMALL_H] > 0

    # --- вся пластина в пределах [z_min, z_max] ---
    z_min, z_max = -0.5 * nz * H, 0.5 * nz * H
    for p in placements:
        half = H if p.kind == BlockKind.DOUBLE_2H else 0.5 * H
        zc = p.center[2]
        if p.kind == BlockKind.HALF_H:
            # центр placement — это центр виртуального двойного, не геометрический центр
            z_lo = zc if p.half_sign >= 0 else zc - H
            z_hi = zc + H if p.half_sign >= 0 else zc
        else:
            z_lo, z_hi = zc - half, zc + half
        assert z_lo >= z_min - 1e-4 and z_hi <= z_max + 1e-4, (p, z_lo, z_hi)
    print(f"  все z в [{z_min:.3f}, {z_max:.3f}] ✓")

    # --- шахматка: цвет клетки ↔ тип ---
    for p in placements:
        black = (p.i + p.j) % 2 == 0
        if black:
            assert p.kind in (BlockKind.DOUBLE_2H, BlockKind.HALF_H), (p.i, p.j, p.kind)
        else:
            assert p.kind in (BlockKind.LARGE_H, BlockKind.SMALL_H), (p.i, p.j, p.kind)
    print("  цвет клетки ↔ тип блока ✓")

    # --- двойные: одна ориентация, сдвиг чётных рядов на H (на одном уровне m) ---
    doubles = [p for p in placements if p.kind == BlockKind.DOUBLE_2H]
    even_d = sorted({p.center[2] for p in doubles if p.j % 2 == 0})
    odd_d = sorted({p.center[2] for p in doubles if p.j % 2 == 1})
    if even_d and odd_d:
        # уровень m: z = z_min + H + 2Hm (+H у чётных)
        z0_even = min(even_d)  # m=0, чётный ряд
        z0_odd = min(odd_d)    # m=0, нечётный ряд
        shift = abs(z0_even - z0_odd)
        print(f"  DOUBLE m=0: even rows z={z0_even:.3f}, odd rows z={z0_odd:.3f}, Δ={shift:.3f}")
        assert abs(shift - H) < 1e-4, f"even/odd ряды должны отличаться на H, получено {shift}"
        print(f"  сдвиг рядов = H ✓")
        # период 2H внутри ряда
        for lst in (even_d, odd_d):
            diffs = [abs(lst[k + 1] - lst[k]) for k in range(len(lst) - 1)]
            assert all(abs(d - 2 * H) < 1e-4 for d in diffs), f"период по Z должен быть 2H: {lst}"

    # --- белые: без сдвига ряда, чередование LARGE/SMALL по k ---
    white = [p for p in placements if (p.i + p.j) % 2 == 1]
    for w in white:
        k = int(round((w.center[2] - z_min - 0.5 * H) / H))
        assert 0 <= k < nz, (w, k)
        expect = BlockKind.LARGE_H if (k % 2 == 0) else BlockKind.SMALL_H
        assert w.kind == expect, (w, k, expect)
    print("  чередование LARGE/SMALL по k ✓")

    # --- HALF только на торцах и правильной ориентации ---
    for p in placements:
        if p.kind != BlockKind.HALF_H:
            continue
        zc = p.center[2]
        if np.isclose(zc, z_min, atol=1e-4):
            assert p.half_sign == 1, ("снизу нужна верхняя половина блока под пластиной", p)
        elif np.isclose(zc, z_max, atol=1e-4):
            assert p.half_sign == -1, ("сверху нужна нижняя половина блока над пластиной", p)
        else:
            raise AssertionError(("HALF должна быть у торца", p))
    print("  HALF на торцах и ориентированы наружу ✓")

    print("✅ test_placement пройден")


# Обратная совместимость со старым именем в ноутбуке
def test_placement_3x3x3() -> None:
    test_placement()


def _block_z_span(p, H: float) -> tuple[float, float]:
    """Абсолютный интервал по Z для блока placement.

    Для HALF center — это центр ВИРТУАЛЬНОГО двойного (стык половинок),
    поэтому реальный интервал зависит от half_sign. Всё в абсолютных единицах.
    """
    zc = float(p.center[2])
    if p.kind == BlockKind.DOUBLE_2H:
        return zc - H, zc + H
    if p.kind == BlockKind.HALF_H:
        if p.half_sign >= 0:
            return zc, zc + H
        return zc - H, zc
    return zc - 0.5 * H, zc + 0.5 * H


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float], eps: float) -> float:
    return min(a[1], b[1]) - max(a[0], b[0]) - eps


def test_black_column(
    nx_bl: int = 2,
    ny_bl: int = 2,
    nz: int = 6,
    H: float = 10.0,
    L: float = 20.0,
) -> None:
    """Юнит: одна чёрная колонка по Z — покрытие без наложений."""
    print("=" * 60)
    print("Юнит: чёрная колонка по Z")
    print("=" * 60)

    config = Tetra4Config(H=H, L=L)
    placements = generate_placements(nx_bl, ny_bl, nz, config)
    z_min, z_max = -0.5 * nz * H, 0.5 * nz * H
    eps = H * 1e-6

    blacks = {}
    for p in placements:
        if (p.i + p.j) % 2 == 0:
            blacks.setdefault((p.i, p.j), []).append(p)

    assert blacks, "нет чёрных клеток"
    for (i, j), col in blacks.items():
        spans = sorted((_block_z_span(p, H), p) for p in col)
        even_row = j % 2 == 0
        kinds = [p.kind for p in col]
        print(f"  клетка ({i},{j}) even_row={even_row}: {[p.kind.value for p in col]}")

        if even_row:
            assert kinds.count(BlockKind.HALF_H) == 2, (i, j, kinds)
            assert BlockKind.DOUBLE_2H in kinds
        else:
            assert BlockKind.HALF_H not in kinds, (i, j, "нечётный ряд не должен иметь HALF")
            assert all(p.kind == BlockKind.DOUBLE_2H for p in col)

        for (s1, p1), (s2, p2) in zip(spans, spans[1:]):
            gap = s2[0] - s1[1]
            assert abs(gap) <= eps, f"дыра/нахлёст в ({i},{j}): {s1} → {s2} gap={gap} {p1.kind}/{p2.kind}"
        assert abs(spans[0][0][0] - z_min) <= eps, (i, j, spans[0])
        assert abs(spans[-1][0][1] - z_max) <= eps, (i, j, spans[-1])

    print("✅ test_black_column пройден")


def test_white_column(
    nx_bl: int = 2,
    ny_bl: int = 2,
    nz: int = 6,
    H: float = 10.0,
    L: float = 20.0,
) -> None:
    """Юнит: белая колонка — nz блоков H, без сдвига ряда, LARGE/SMALL по k."""
    print("=" * 60)
    print("Юнит: белая колонка по Z")
    print("=" * 60)

    config = Tetra4Config(H=H, L=L)
    placements = generate_placements(nx_bl, ny_bl, nz, config)
    z_min = -0.5 * nz * H
    eps = H * 1e-6

    whites = {}
    for p in placements:
        if (p.i + p.j) % 2 == 1:
            whites.setdefault((p.i, p.j), []).append(p)

    assert whites, "нет белых клеток"
    for (i, j), col in whites.items():
        col = sorted(col, key=lambda p: p.center[2])
        assert len(col) == nz, (i, j, len(col))
        for k, p in enumerate(col):
            expect_z = z_min + (k + 0.5) * H
            assert abs(p.center[2] - expect_z) <= eps, (i, j, k, p.center[2], expect_z)
            expect = BlockKind.LARGE_H if k % 2 == 0 else BlockKind.SMALL_H
            assert p.kind == expect, (i, j, k, p.kind, expect)
            lo, hi = _block_z_span(p, H)
            assert abs((hi - lo) - H) <= eps
        for a, b in zip(col, col[1:]):
            gap = _block_z_span(b, H)[0] - _block_z_span(a, H)[1]
            assert abs(gap) <= eps, (i, j, gap)

    print("✅ test_white_column пройден")


def test_same_cell_no_z_overlap(
    nx_bl: int = 3,
    ny_bl: int = 3,
    nz: int = 6,
    H: float = 10.0,
    L: float = 20.0,
) -> None:
    """Юнит: в одной клетке интервалы Z не налезают (касание ок)."""
    from collections import defaultdict
    from itertools import combinations

    print("=" * 60)
    print("Юнит: нет Z-нахлёста в одной клетке")
    print("=" * 60)

    config = Tetra4Config(H=H, L=L)
    placements = generate_placements(nx_bl, ny_bl, nz, config)
    by_cell = defaultdict(list)
    for p in placements:
        by_cell[(p.i, p.j)].append(p)

    bad = []
    for cell, col in by_cell.items():
        for p1, p2 in combinations(col, 2):
            ov = _intervals_overlap(_block_z_span(p1, H), _block_z_span(p2, H), H * 1e-4)
            if ov > 0:
                bad.append((cell, p1.kind, p2.kind, ov))

    if bad:
        for rec in bad[:12]:
            print("  ", rec)
        raise AssertionError(f"Z-нахлёст в клетке: {len(bad)} пар")

    print("✅ test_same_cell_no_z_overlap пройден")


def _aabb(poly) -> tuple[np.ndarray, np.ndarray]:
    v = poly.vertices
    return v.min(axis=0), v.max(axis=0)


def _aabb_overlap(a_min, a_max, b_min, b_max, eps: float) -> bool:
    return bool(np.all(a_max > b_min + eps) and np.all(b_max > a_min + eps))


def _point_strictly_inside(poly, pt: np.ndarray, eps: float) -> bool:
    pt = np.asarray(pt, dtype=float)
    for p, n in zip(poly.plane_points, poly.plane_normals):
        n = np.asarray(n, dtype=float)
        nlen = np.linalg.norm(n)
        if nlen < 1e-15:
            continue
        if np.dot(n / nlen, pt - np.asarray(p, dtype=float)) > -eps:
            return False
    return True


def test_no_overlap(
    nx_bl: int = 2,
    ny_bl: int = 2,
    nz: int = 4,
    H: float = 10.0,
    L: float = 20.0,
    alpha_deg: float = 25.0,
) -> None:
    """
    Внутренности ConvexPolyhedron не пересекаются.

    Соседи МОГУТ делить диапазон Z — так устроен слой (тетраэдрический стык).
    Критерий: нет точки, строго лежащей внутри обоих тел.
    """
    from itertools import combinations

    print("=" * 60)
    print(f"Тест: внутренности полиэдров {nx_bl}×{ny_bl}×{nz}")
    print("=" * 60)

    config = Tetra4Config(H=H, L=L, alpha_deg=alpha_deg)
    blocks, placements = Tetra4LayerGen(nx_bl, ny_bl, nz, config)
    print(f"  блоков: {len(blocks)}")

    eps = max(config.tolerance * 20.0, 1e-3 * H)
    aabbs = [_aabb(b) for b in blocks]
    bad = []

    for i, j in combinations(range(len(blocks)), 2):
        if not _aabb_overlap(*aabbs[i], *aabbs[j], eps):
            continue
        a, b = blocks[i], blocks[j]
        samples = list(a.vertices) + list(b.vertices)
        samples.append(a.vertices.mean(axis=0))
        samples.append(b.vertices.mean(axis=0))
        # немного внутрь от вершин к центроиду
        ca, cb = a.vertices.mean(axis=0), b.vertices.mean(axis=0)
        for v in a.vertices:
            samples.append(0.85 * v + 0.15 * ca)
        for v in b.vertices:
            samples.append(0.85 * v + 0.15 * cb)
        hit = None
        for pt in samples:
            if _point_strictly_inside(a, pt, eps) and _point_strictly_inside(b, pt, eps):
                hit = pt
                break
        if hit is not None:
            p1, p2 = placements[i], placements[j]
            bad.append((p1.kind.value, p2.kind.value, tuple(np.round(p1.center, 3)), tuple(np.round(p2.center, 3)), np.round(hit, 3)))

    if bad:
        print(f"  ⚠️ пересечение внутренностей: {len(bad)} пар")
        for rec in bad[:12]:
            print("   ", rec)
        raise AssertionError(f"Обнаружено пересечение внутренностей: {len(bad)} пар")

    print("  нет общей внутренней точки ✓")
    print("✅ test_no_overlap пройден")


def test_layer_gen(
    is_visualized: bool = True,
    interactive_backend: bool = False,
    nx_bl: int = 3,
    ny_bl: int = 3,
    nz: int = 6,
    H: float = 10.0,
    L: float = 20.0,
    alpha_deg: float = 25.0,
    viz_scale: float = 0.99,
) -> None:
    print("=" * 60)
    print(f"Тест: слой Tetra4LayerGen {nx_bl}×{ny_bl} двойных × {nz} слоёв")
    print("=" * 60)

    config = Tetra4Config(H=H, L=L, alpha_deg=alpha_deg)
    blocks, placements = Tetra4LayerGen(nx_bl, ny_bl, nz, config)
    print(f"✓ блоков: {len(blocks)}")
    assert len(blocks) == len(placements)
    assert all(b.faces for b in blocks)

    stl_path = _OUT / f"layer_gen_{nx_bl}x{ny_bl}x{nz}.stl"
    export_blocks_stl(blocks, str(stl_path), solid_name=f"layer_{nx_bl}x{ny_bl}x{nz}")
    print(f"✓ STL: {stl_path.name} ({stl_path.stat().st_size} байт)")

    if interactive_backend:
        backend = _try_interactive_backend()
        if backend:
            print(f"  Бэкенд: {backend}")

    style = replace(VIZ, display_scale=viz_scale)
    print(f"  viz display_scale={style.display_scale} (только картинка, STL полный размер)")

    png_path = _OUT / f"layer_gen_{nx_bl}x{ny_bl}x{nz}.png"
    visualize_layer(
        blocks,
        placements,
        show=is_visualized,
        save_path=png_path,
        title=f"trunc-tetra-4 layer {nx_bl}×{ny_bl} doubles × {nz} layers  scale={viz_scale}",
        style=style,
    )
    print(f"✓ PNG: {png_path.name}")
    print("✅ test_layer_gen пройден")


def test_reference_block_joint(
    is_visualized: bool = True,
    interactive_backend: bool = False,
    H: float = 3.0,
    L: float = 2.0,
    alpha_deg: float = 25.0,
    viz_scale: float = 0.97,
) -> None:
    """Эталонный узел как на физическом образце.

    Один DOUBLE_2H на чёрной клетке окружён четырьмя белыми клетками.
    В каждой белой клетке видны два блока H: SMALL снизу и LARGE сверху.
    Итого: 1 DOUBLE + 4 SMALL + 4 LARGE.
    """
    print("=" * 60)
    print("Тест: эталонный узел блоков")
    print("=" * 60)

    nx_bl = ny_bl = 3
    nz = 6
    config = Tetra4Config(H=H, L=L, alpha_deg=alpha_deg)
    blocks, placements = Tetra4LayerGen(nx_bl, ny_bl, nz, config)

    ci = cj = 2
    target_z = -H
    selected: list[tuple] = []
    for block, p in zip(blocks, placements):
        is_center_double = (
            p.kind == BlockKind.DOUBLE_2H
            and p.i == ci
            and p.j == cj
            and np.isclose(p.center[2], target_z)
        )
        is_adjacent_white = (
            p.kind in (BlockKind.LARGE_H, BlockKind.SMALL_H)
            and abs(p.i - ci) + abs(p.j - cj) == 1
            and np.isclose(abs(p.center[2] - target_z), 0.5 * H)
        )
        if is_center_double or is_adjacent_white:
            selected.append((block, p))

    sample_blocks = [item[0] for item in selected]
    sample_placements = [item[1] for item in selected]
    counts = Counter(p.kind for p in sample_placements)

    assert len(sample_blocks) == 9, f"ожидается 9 блоков узла, получено {len(sample_blocks)}"
    assert counts == Counter(
        {
            BlockKind.DOUBLE_2H: 1,
            BlockKind.SMALL_H: 4,
            BlockKind.LARGE_H: 4,
        }
    ), counts

    center = next(p for p in sample_placements if p.kind == BlockKind.DOUBLE_2H)
    for p in sample_placements:
        if p.kind not in (BlockKind.LARGE_H, BlockKind.SMALL_H):
            continue
        dxy = np.abs(p.center[:2] - center.center[:2])
        assert np.isclose(dxy.sum(), config.cell_pitch), (p, dxy, config.cell_pitch)
        expected_dz = -0.5 * H if p.kind == BlockKind.SMALL_H else 0.5 * H
        assert np.isclose(p.center[2] - center.center[2], expected_dz), (p, expected_dz)

    # Уточнение инженера: верхний торец — НИЖНЯЯ половина виртуального
    # двойного над пластиной; нижний торец — ВЕРХНЯЯ половина под пластиной.
    z_min, z_max = -0.5 * nz * H, 0.5 * nz * H
    top_halves = [
        (block, p) for block, p in zip(blocks, placements)
        if p.kind == BlockKind.HALF_H and np.isclose(p.center[2], z_max)
    ]
    bottom_halves = [
        (block, p) for block, p in zip(blocks, placements)
        if p.kind == BlockKind.HALF_H and np.isclose(p.center[2], z_min)
    ]
    assert top_halves and all(p.half_sign == -1 for _, p in top_halves)
    assert bottom_halves and all(p.half_sign == 1 for _, p in bottom_halves)

    print(f"  H={H}, L={L}, фактический XY-шаг={config.cell_pitch:.6g}")
    print("  состав: 1 DOUBLE_2H + 4 SMALL_H + 4 LARGE_H ✓")
    print("  белые центры: ±H/2 по Z и один XY-шаг от DOUBLE ✓")
    print("  торцевые HALF ориентированы наружу ✓")

    tag = "reference_block_joint"
    stl_path = _OUT / f"{tag}.stl"
    export_blocks_stl(sample_blocks, str(stl_path), solid_name=tag)

    if interactive_backend:
        backend = _try_interactive_backend()
        if backend:
            print(f"  Бэкенд: {backend}")

    png_path = _OUT / f"{tag}.png"
    style = replace(VIZ, display_scale=viz_scale)
    visualize_layer(
        sample_blocks,
        sample_placements,
        show=is_visualized,
        save_path=png_path,
        title=(
            "reference block joint: "
            "1 DOUBLE + 4 SMALL below + 4 LARGE above"
        ),
        style=style,
    )
    print(f"✓ STL: {stl_path}")
    print(f"✓ PNG: {png_path}")

    # Отдельно показываем исправленный верхний торец из диалога:
    # одна нижняя HALF виртуального двойного над пластиной + 4 соседних SMALL.
    cap_selected: list[tuple] = []
    for block, p in zip(blocks, placements):
        is_top_half = (
            p.kind == BlockKind.HALF_H
            and p.i == ci
            and p.j == cj
            and np.isclose(p.center[2], z_max)
            and p.half_sign == -1
        )
        is_top_neighbor = (
            p.kind == BlockKind.SMALL_H
            and abs(p.i - ci) + abs(p.j - cj) == 1
            and np.isclose(p.center[2], z_max - 0.5 * H)
        )
        if is_top_half or is_top_neighbor:
            cap_selected.append((block, p))

    cap_blocks = [item[0] for item in cap_selected]
    cap_placements = [item[1] for item in cap_selected]
    assert Counter(p.kind for p in cap_placements) == Counter(
        {BlockKind.HALF_H: 1, BlockKind.SMALL_H: 4}
    )
    cap_stl = _OUT / "reference_end_cap_joint.stl"
    cap_png = _OUT / "reference_end_cap_joint.png"
    export_blocks_stl(cap_blocks, str(cap_stl), solid_name="reference_end_cap_joint")
    visualize_layer(
        cap_blocks,
        cap_placements,
        show=False,
        save_path=cap_png,
        title="correct top end cap: lower HALF above + 4 SMALL",
        style=style,
    )
    print(f"✓ STL торца: {cap_stl}")
    print(f"✓ PNG торца: {cap_png}")
    print("✅ test_reference_block_joint пройден")


def test_layer_cut(
    is_visualized: bool = True,
    interactive_backend: bool = False,
    nx_bl: int = 2, # 3
    ny_bl: int = 2, # 3
    nz: int = 4, # 6
    H: float = 10.0,
    L: float = 20.0,
    alpha_deg: float = 25.0,
    plane_point: np.ndarray | None = None,
    plane_normal: np.ndarray | None = None,
    keep_negative: bool = True,
) -> None:
    """Слой в разрезе: выкидываем блоки, чей центр по одну сторону от плоскости.

    По умолчанию косая плоскость n=(1,1,0.35) через центр пластины —
    как половина параллелепипеда, срезанная под углом.
    """
    print("=" * 60)
    print(f"Тест: разрез слоя {nx_bl}×{ny_bl}×{nz}")
    print("=" * 60)

    if plane_point is None:
        plane_point = np.zeros(3)
    if plane_normal is None:
        plane_normal = np.array([1.0, 1.0, 0.35])

    config = Tetra4Config(H=H, L=L, alpha_deg=alpha_deg)
    blocks, placements = Tetra4LayerGen(nx_bl, ny_bl, nz, config)
    kept_b, kept_p = filter_blocks_by_plane(
        blocks,
        placements,
        plane_point=plane_point,
        plane_normal=plane_normal,
        keep_negative=keep_negative,
    )
    dropped = len(blocks) - len(kept_b)
    print(f"  всего: {len(blocks)}, осталось: {len(kept_b)}, отброшено: {dropped}")
    print(f"  плоскость: p={np.asarray(plane_point)}, n={np.asarray(plane_normal)}")
    print(f"  сторона: {'n·(c−p) ≤ 0' if keep_negative else 'n·(c−p) ≥ 0'}")

    assert 0 < len(kept_b) < len(blocks), (
        f"разрез должен оставить часть блоков, не все и не ноль: kept={len(kept_b)}/{len(blocks)}"
    )

    tag = f"layer_cut_{nx_bl}x{ny_bl}x{nz}"
    stl_path = _OUT / f"{tag}.stl"
    export_blocks_stl(kept_b, str(stl_path), solid_name=tag)
    print(f"✓ STL: {stl_path.name} ({stl_path.stat().st_size} байт)")

    if interactive_backend:
        backend = _try_interactive_backend()
        if backend:
            print(f"  Бэкенд: {backend}")

    png_path = _OUT / f"{tag}.png"
    visualize_layer(
        kept_b,
        kept_p,
        show=is_visualized,
        save_path=png_path,
        title=f"trunc-tetra-4 cut {nx_bl}×{ny_bl}×{nz}  n={np.round(plane_normal, 2)}",
        cut_plane=(plane_point, plane_normal),
    )
    print(f"✓ PNG: {png_path.name}")
    print("✅ test_layer_cut пройден")


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="trunc-tetra-4 geom tests")
    p.add_argument(
        "--steps",
        default="0",
        help="через запятую: 0 примитив, 1 четыре типа, 2 половинки, 3 placement, "
        "4 слой, 5 внутренности, 6 чёрная колонка, 7 белая колонка, 8 Z в клетке, 9 разрез",
    )
    p.add_argument(
        "--test",
        choices=("reference-block-joint",),
        default=None,
        help="именованный тест; reference-block-joint — эталонный узел как на фото",
    )
    p.add_argument("--window", "-w", action="store_true", help="интерактивное окно matplotlib")
    p.add_argument("--H", type=float, default=10.0)
    p.add_argument("--L", type=float, default=20.0)
    p.add_argument("--alpha", type=float, default=25.0)
    p.add_argument("--nx", type=int, default=3, help="двойных блоков по X (для шагов 3–4)")
    p.add_argument("--ny", type=int, default=3)
    p.add_argument("--nz", type=int, default=6, help="чётное число слоёв H")
    p.add_argument("--cut-nx", type=float, default=1.0, help="нормаль режущей плоскости, X")
    p.add_argument("--cut-ny", type=float, default=1.0, help="нормаль режущей плоскости, Y")
    p.add_argument("--cut-nz", type=float, default=0.35, help="нормаль режущей плоскости, Z")
    p.add_argument(
        "--cut-keep-positive",
        action="store_true",
        help="оставить n·(c−p) ≥ 0 вместо ≤ 0",
    )
    p.add_argument(
        "--viz-scale",
        type=float,
        default=0.99,
        help="линейный масштаб блоков только для картинки шага 4 (0.99 = −1%%)",
    )
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    steps = {s.strip() for s in a.steps.split(",") if s.strip()}

    if a.test == "reference-block-joint":
        test_reference_block_joint(
            is_visualized=a.window,
            interactive_backend=a.window,
            H=a.H,
            L=a.L,
            alpha_deg=a.alpha,
            viz_scale=a.viz_scale,
        )
        return

    if "0" in steps:
        test_truncated_tetrahedron(show=True, interactive_backend=a.window)
    if "1" in steps:
        test_four_block_types(show=True, interactive_backend=a.window)
    if "2" in steps:
        test_halves_match_double(H=a.H, L=a.L, alpha_deg=a.alpha)
    if "3" in steps:
        test_placement(nx_bl=a.nx, ny_bl=a.ny, nz=a.nz, H=a.H, L=a.L, alpha_deg=a.alpha)
    if "6" in steps:
        test_black_column(nx_bl=a.nx, ny_bl=a.ny, nz=a.nz, H=a.H, L=a.L)
    if "7" in steps:
        test_white_column(nx_bl=a.nx, ny_bl=a.ny, nz=a.nz, H=a.H, L=a.L)
    if "8" in steps:
        test_same_cell_no_z_overlap(nx_bl=a.nx, ny_bl=a.ny, nz=a.nz, H=a.H, L=a.L)
    if "5" in steps:
        test_no_overlap(nx_bl=a.nx, ny_bl=a.ny, nz=a.nz, H=a.H, L=a.L, alpha_deg=a.alpha)
    if "4" in steps:
        test_layer_gen(
            is_visualized=a.window,
            interactive_backend=a.window,
            nx_bl=a.nx,
            ny_bl=a.ny,
            nz=a.nz,
            H=a.H,
            L=a.L,
            alpha_deg=a.alpha,
            viz_scale=a.viz_scale,
        )
    if "9" in steps:
        test_layer_cut(
            is_visualized=a.window,
            interactive_backend=a.window,
            nx_bl=a.nx,
            ny_bl=a.ny,
            nz=a.nz,
            H=a.H,
            L=a.L,
            alpha_deg=a.alpha,
            plane_normal=np.array([a.cut_nx, a.cut_ny, a.cut_nz]),
            keep_negative=not a.cut_keep_positive,
        )


if __name__ == "__main__":
    main()
