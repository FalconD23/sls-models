'''
элементарные функции для прототипирования bt_octs
и сверка в freecad
'''

import numpy as np
import sys
import os
from pathlib import Path
# Add parent directory to path

sys.path.insert(0, str(Path(__file__).parent.parent))
from geometry.convex_polyhedron import ConvexPolyhedron


def create_truncated_octahedron(
    a: float = 8.0, 
    b: float = 8.0, 
    c: float = 8.0, 
    alpha_deg: float = 0.0,
    center: np.ndarray = np.array([0, 0, 0])
) -> tuple[ConvexPolyhedron, dict]:
    """
    Create a standard truncated octahedron.
    
    Truncated octahedron has:
    - 14 faces: 6 squares + 8 hexagons
    - 24 vertices
    - 36 edges
    
    Args:
        a: Distance along X axis
        b: Distance along Y axis
        c: Distance along Z axis
        alpha_deg: Bevel angle in degrees
        
    Returns:
        Tuple of (ConvexPolyhedron instance, parameters dict)
    """
    alpha = np.radians(alpha_deg)
    tgalpha = np.tan(alpha)
    
    # Для правильного усеченного октаэдра:
    # - В каждой вершине сходятся 2 шестиугольные + 1 квадратная грань
    # - Вершины: всевозможные перестановки (0; ±1; ±2) для ребра длины √2
    #
    # Для нашей геометрии с квадратными гранями на x=±a, y=±a, z=±a:
    # Нужно найти правильное расстояние до шестиугольных граней так,
    # чтобы каждая вершина была на пересечении 1 квадратной + 2 шестиугольных граней
    #
    # Для вершины на квадратной грани x=a с координатами (a, y, z):
    # Она должна быть на пересечении: sq_x_pos + hex1 + hex2
    # Для правильного усеченного октаэдра: используем стандартное соотношение
    # Для усеченного октаэдра с квадратными гранями на расстоянии a:
    # Шестиугольные грани находятся на расстоянии a*sqrt(2)/2 от центра
    # Это обеспечивает правильную структуру вершин
    # hex_center_dist = a * np.sqrt(2) / 2.0
    
    plane_points = []
    plane_normals = []
    
    # 6 square faces (parallel to coordinate planes)
    # Centers of square faces are on coordinate axes
    # Normals point OUTWARD from center
    
    # Face 0: x = a (right square face)
    plane_points.append(np.array([a, 0, 0]) + center)  # Center of square face
    plane_normals.append(np.array([1, 0, tgalpha]))  # Normal outward
    
    # Face 1: x = -a (left square face)
    plane_points.append(np.array([-a, 0, 0]) + center)
    plane_normals.append(np.array([-1, 0, tgalpha]))  # Normal outward
    
    # Face 2: y = a (front square face)
    plane_points.append(np.array([0, b, 0]) + center)
    plane_normals.append(np.array([0, 1, -tgalpha]))  # Normal outward
    
    # Face 3: y = -a (back square face)
    plane_points.append(np.array([0, -b, 0]) + center)
    plane_normals.append(np.array([0, -1, -tgalpha]))  # Normal outward
    
    # Face 4: z = a (top square face)
    plane_points.append(np.array([0, 0, c]) + center)
    plane_normals.append(np.array([0, 0, 1]))  # Normal outward
    
    # Face 5: z = -a (bottom square face)
    plane_points.append(np.array([0, 0, -c]) + center)
    plane_normals.append(np.array([0, 0, -1]))  # Normal outward
    
    # 8 hexagonal faces
    # Hexagonal faces have centers at hex_center_dist from origin
    # along directions toward 8 octants
    # Normals point OUTWARD (away from origin, toward octants)
    
    hex_directions = [
        np.array([1, 1, 1]),      # Octant (+++)
        np.array([1, 1, -1]),     # Octant (++-)
        np.array([1, -1, 1]),     # Octant (+-+)
        np.array([1, -1, -1]),    # Octant (+--)
        np.array([-1, 1, 1]),     # Octant (-++)
        np.array([-1, 1, -1]),    # Octant (-+-)
        np.array([-1, -1, 1]),    # Octant (--+)
        np.array([-1, -1, -1]),   # Octant (---)
    ]
    
    for hex_dir in hex_directions:
        # Normalize direction
        hex_dir_normalized = hex_dir / np.linalg.norm(hex_dir)
        
        # Center of hexagonal face (point on the face)
        # hex_center = hex_dir_normalized * hex_center_dist
        # hex_center = (a + b + c) / 6 * hex_dir
        hex_center = (1/2) * (np.array([a, b, c]) * hex_dir) + center
        
        # Normal points in same direction as center (outward)
        plane_points.append(hex_center)
        plane_normals.append(hex_dir_normalized)
    
    # ============================================================================
    # ИМЕНОВАННЫЕ ИНДЕКСЫ ПЛОСКОСТЕЙ
    # ============================================================================
    # Квадратные грани (6 штук):
    sq_x_pos = 0  # x = +a, правая грань
    sq_x_neg = 1  # x = -a, левая грань
    sq_y_pos = 2  # y = +a, передняя грань
    sq_y_neg = 3  # y = -a, задняя грань
    sq_z_pos = 4  # z = +a, верхняя грань
    sq_z_neg = 5  # z = -a, нижняя грань
    
    # Шестиугольные грани (8 штук):
    hex_ppp = 6   # +++, нормаль в направлении (1,1,1)
    hex_ppn = 7   # ++-, нормаль в направлении (1,1,-1)
    hex_pnp = 8   # +-+, нормаль в направлении (1,-1,1)
    hex_pnn = 9   # +--, нормаль в направлении (1,-1,-1)
    hex_npp = 10  # -++, нормаль в направлении (-1,1,1)
    hex_npn = 11  # -+-, нормаль в направлении (-1,1,-1)
    hex_nnp = 12  # --+, нормаль в направлении (-1,-1,1)
    hex_nnn = 13  # ---, нормаль в направлении (-1,-1,-1)
    
    # ============================================================================
    # ОПРЕДЕЛЕНИЕ ТРОЕК ПЛОСКОСТЕЙ ДЛЯ 24 ВЕРШИН УСЕЧЕННОГО ОКТАЭДРА
    # ============================================================================
    #
    # Усеченный октаэдр имеет 24 вершины
    # В каждой вершине сходятся: 2 шестиугольные + 1 квадратная грань
    # Значит, каждая вершина - пересечение: 1 квадратная грань + 2 шестиугольные грани
    #
    # Для 6 квадратных граней, каждая с 4 углами = 24 вершины
    # Каждая вершина на квадратной грани - пересечение этой грани + 2 шестиугольных граней
    #
    # ============================================================================
    
    vertex_triplets = []
    
    # ----------------------------------------------------------------------------
    # ОПРЕДЕЛЕНИЕ ВЕРШИН ДЛЯ 6 КВАДРАТНЫХ ГРАНЕЙ
    # Каждая квадратная грань имеет 4 вершины по углам
    # Каждая вершина = пересечение этой квадратной грани + 2 шестиугольные грани
    # Итого: 6 граней × 4 угла = 24 вершины
    # ----------------------------------------------------------------------------
    
    # КВАДРАТНАЯ ГРАНЬ 1: x=a (sq_x_pos) - правая грань
    # 4 угла квадрата на пересечении sq_x_pos + 2 шестиугольных граней
    # Каждый угол соответствует паре октантов с положительным x
    vertex_triplets.append((sq_x_pos, hex_ppp, hex_ppn))  # Угол: пересечение hex_ppp и hex_ppn
    vertex_triplets.append((sq_x_pos, hex_ppn, hex_pnn))  # Угол: пересечение hex_ppn и hex_pnn
    vertex_triplets.append((sq_x_pos, hex_pnp, hex_pnn))  # Угол: пересечение hex_pnp и hex_pnn
    


    vertex_triplets.append((sq_x_neg, hex_npp, hex_npn))  # Угол: пересечение hex_npp и hex_npn
    vertex_triplets.append((sq_x_neg, hex_npn, hex_nnn))  # Угол: пересечение hex_npn и hex_nnn
    vertex_triplets.append((sq_x_neg, hex_nnp, hex_nnn))  # Угол: пересечение hex_nnp и hex_nnn
    


    vertex_triplets.append((sq_y_pos, hex_ppp, hex_ppn))  # Угол: пересечение hex_ppp и hex_ppn
    vertex_triplets.append((sq_y_pos, hex_ppp, hex_npp))  # Угол: пересечение hex_ppp и hex_npp
    vertex_triplets.append((sq_y_pos, hex_npp, hex_npn))  # Угол: пересечение hex_npp и hex_npn
    


    vertex_triplets.append((sq_y_neg, hex_pnp, hex_pnn))  # Угол: пересечение hex_pnp и hex_pnn
    vertex_triplets.append((sq_y_neg, hex_pnp, hex_nnp))  # Угол: пересечение hex_pnp и hex_nnp
    vertex_triplets.append((sq_y_neg, hex_nnp, hex_nnn))  # Угол: пересечение hex_nnp и hex_nnn
    


    vertex_triplets.append((sq_z_pos, hex_ppp, sq_x_pos))
    vertex_triplets.append((sq_z_pos, hex_pnp, sq_x_pos))
    vertex_triplets.append((sq_z_pos, hex_nnp, sq_x_neg))  # Угол: пересечение hex_pnp и hex_nnp
    vertex_triplets.append((sq_z_pos, hex_npp, sq_x_neg))  # Угол: пересечение hex_npp и hex_nnp
    vertex_triplets.append((sq_z_pos, hex_ppp, hex_npp))  # Угол: пересечение hex_ppp и hex_npp
    vertex_triplets.append((sq_z_pos, hex_pnp, hex_nnp)) 


    vertex_triplets.append((sq_z_neg, hex_ppn, sq_y_pos))  # Угол: пересечение hex_ppn и hex_pnn
    vertex_triplets.append((sq_z_neg, hex_npn, sq_y_pos))  # Угол: пересечение hex_ppn и hex_npn
    vertex_triplets.append((sq_z_neg, hex_pnn, sq_y_neg))  # Угол: пересечение hex_pnn и hex_nnn
    vertex_triplets.append((sq_z_neg, hex_nnn, sq_y_neg))  # Угол: пересечение hex_npn и hex_nnn
    vertex_triplets.append((sq_z_neg, hex_ppn, hex_pnn))  # Угол: пересечение hex_ppn и hex_pnn
    vertex_triplets.append((sq_z_neg, hex_npn, hex_nnn))



    # ----------------------------------------------------------------------------
    # УДАЛЕНИЕ ДУБЛИКАТОВ: вычисляем вершины и оставляем только уникальные
    # ----------------------------------------------------------------------------
    unique_triplets = []
    seen_vertices = []
    vertex_names = []  # Для отладки - названия вершин
    
    for triplet in vertex_triplets:
        # Вычисляем вершину для этой тройки
        temp_poly = ConvexPolyhedron(
            plane_points=plane_points,
            plane_normals=plane_normals,
            vertex_triplets=[triplet],
            tolerance=1e-5
        )
        temp_vertices = temp_poly.build_vertices()
        
        if len(temp_vertices) > 0:
            vertex = temp_vertices[0]
            
            # Проверяем, уникальна ли эта вершина
            is_unique = True
            for i, seen_vertex in enumerate(seen_vertices):
                if np.allclose(vertex, seen_vertex, atol=1e-4):
                    is_unique = False
                    # Для отладки: если вершина дублируется, используем то же имя
                    break
            
            if is_unique:
                unique_triplets.append(triplet)
                seen_vertices.append(vertex)
                # Определяем имя вершины для отладки
                v_str = f"({vertex[0]:.1f}, {vertex[1]:.1f}, {vertex[2]:.1f})"
                vertex_names.append(f"vertex_{len(unique_triplets)}: {v_str}")
    
    vertex_triplets = unique_triplets
    
    # Выводим информацию о вершинах для отладки
    if len(vertex_triplets) < 24:
        print(f"⚠️ Внимание: получено только {len(vertex_triplets)} уникальных вершин вместо 24")
        print("   Это означает, что некоторые тройки дают одинаковые вершины")
        print("   Уникальные вершины:")
        for name in vertex_names[:10]:  # Показываем первые 10
            print(f"     {name}")
    
    polyhedron = ConvexPolyhedron(
        plane_points=plane_points,
        plane_normals=plane_normals,
        vertex_triplets=vertex_triplets,
        tolerance=1e-5
    )
    
    params = {'a': a, 'b': b, 'c': c, 'alpha_deg': alpha_deg}
    
    return polyhedron, params





def translate_polyhedron(polyhedron: ConvexPolyhedron, offset: np.ndarray) -> ConvexPolyhedron:
    """
    Create a translated copy of polyhedron.
    
    Args:
        polyhedron: Source polyhedron
        offset: Translation vector [dx, dy, dz]
        
    Returns:
        New ConvexPolyhedron translated by offset
    """
    translated_plane_points = [pt + offset for pt in polyhedron.plane_points]
    
    return ConvexPolyhedron(
        plane_points=translated_plane_points,
        plane_normals=polyhedron.plane_normals.copy(),
        vertex_triplets=polyhedron.vertex_triplets.copy(),
        tolerance=polyhedron.tolerance
    )


def rotate_polyhedron(
    polyhedron: ConvexPolyhedron, 
    axis: np.ndarray = np.array([0, 0, 1]), 
    angle_deg: float = 90.0,
    center: np.ndarray = np.array([0, 0, 0])
) -> ConvexPolyhedron:
    """
    Create a rotated copy of polyhedron around given axis.
    
    Uses Rodrigues' rotation formula for 3D rotation.
    
    Args:
        polyhedron: Source polyhedron
        axis: Rotation axis vector [x, y, z] (will be normalized)
        angle_deg: Rotation angle in degrees (default: 90°, counter-clockwise)
        center: Point around which to rotate (default: origin [0, 0, 0])
        
    Returns:
        New ConvexPolyhedron rotated around axis
    """
    # Normalize axis
    axis = np.array(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        raise ValueError("Axis vector cannot be zero")
    axis = axis / axis_norm
    
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    # Rotation center (default: origin)
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.array(center, dtype=float)
    
    def rotate_vector(v: np.ndarray) -> np.ndarray:
        """
        Rotate vector v around axis using Rodrigues' formula.
        
        Formula: v_rot = v*cos(θ) + (axis × v)*sin(θ) + axis*(axis·v)*(1-cos(θ))
        """
        v = np.array(v, dtype=float)
        # Rodrigues' rotation formula
        v_rot = (
            v * cos_theta +
            np.cross(axis, v) * sin_theta +
            axis * np.dot(axis, v) * (1 - cos_theta)
        )
        return v_rot
    
    # Rotate plane points (relative to center)
    rotated_plane_points = []
    for pt in polyhedron.plane_points:
        # Translate to origin, rotate, translate back
        pt_relative = pt - center
        pt_rotated = rotate_vector(pt_relative)
        rotated_plane_points.append(pt_rotated + center)
    
    # Rotate plane normals (vectors, not points)
    rotated_plane_normals = [rotate_vector(normal) for normal in polyhedron.plane_normals]
    
    return ConvexPolyhedron(
        plane_points=rotated_plane_points,
        plane_normals=rotated_plane_normals,
        vertex_triplets=polyhedron.vertex_triplets.copy(),
        tolerance=polyhedron.tolerance
    )




def generate_tessellation_centers(
    nx: int,
    ny: int,
    nz: int,
    a: float,
    b: float,
    c: float
) -> tuple[list[np.ndarray], list[tuple[int, int, int, bool]]]:
    """
    Generate centers for space-filling tessellation with truncated octahedrons.
    
    Усечённый октаэдр заполняет пространство так, что каждый блок имеет 14 соседей:
    - 6 соседей через квадратные грани (по осям ±X, ±Y, ±Z) - расстояние 2a, 2b, 2c
    - 8 соседей через шестиугольные грани (по диагоналям) - расстояние (a, b, c) в направлениях (±1, ±1, ±1)
    
    Структура замощения использует две подрешётки:
    - Основная решётка: центры в (2a*i, 2b*j, 2c*k)
    - Смещённая решётка: центры в (2a*i + a, 2b*j + b, 2c*k + c) относительно основной решётки
    
    Блоки смещённой решётки имеют тот же тип (A/B), что и ближайший блок основной решётки.
    
    Args:
        nx: Number of blocks along X axis (основная решётка)
        ny: Number of blocks along Y axis (основная решётка)
        nz: Number of blocks along Z axis (основная решётка)
        a: Block parameter a (half-size along X)
        b: Block parameter b (half-size along Y)
        c: Block parameter c (half-size along Z)
        
    Returns:
        Tuple of (list of centers, list of (i, j, k, is_offset) indices)
        where is_offset=True для блоков смещённой решётки
    """
    centers = []
    indices = []
    
    # Расстояние между центрами соседних блоков основной решётки
    # Для соприкосновения квадратных граней: расстояние = 2a, 2b, 2c
    step_x = 2.0 * a
    step_y = 2.0 * b
    step_z = 2.0 * c
    
    # Вычисляем общий размер слоя (основная решётка)
    total_size_x = (nx - 1) * step_x if nx > 1 else 0
    total_size_y = (ny - 1) * step_y if ny > 1 else 0
    total_size_z = (nz - 1) * step_z if nz > 1 else 0
    
    # Смещение для центрирования (чтобы центр слоя был в начале координат)
    offset_x = -total_size_x / 2.0
    offset_y = -total_size_y / 2.0
    offset_z = -total_size_z / 2.0
    
    # ========================================================================
    # ОСНОВНАЯ РЕШЁТКА: центры в (2a*i, 2b*j, 2c*k)
    # ========================================================================
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                center_x = offset_x + i * step_x
                center_y = offset_y + j * step_y
                center_z = offset_z + k * step_z
                center = np.array([center_x, center_y, center_z])
                
                centers.append(center)
                indices.append((i, j, k, False))  # False = основная решётка
    
    # ========================================================================
    # СМЕЩЁННАЯ РЕШЁТКА: центры в (2a*i + a, 2b*j + b, 2c*k + c)
    # Эти блоки соприкасаются с блоками основной решётки через шестиугольные грани
    # ========================================================================
    # Для смещённой решётки нужно на 1 больше по каждой оси, чтобы заполнить промежутки
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Смещение на (a, b, c) относительно основной решётки
                center_x = offset_x + i * step_x + a
                center_y = offset_y + j * step_y + b
                center_z = offset_z + k * step_z + c
                center = np.array([center_x, center_y, center_z])
                
                centers.append(center)
                # Для смещённой решётки используем те же индексы (i, j, k), но is_offset=True
                # Тип блока определяется по индексам ближайшего блока основной решётки
                indices.append((i, j, k, True))  # True = смещённая решётка

    return centers, indices


def LayerGen(
    nx: int,
    ny: int,
    nz: int,
    a: float,
    b: float,
    c: float,
    alpha_deg: float
) -> list[ConvexPolyhedron]:
    """
    Generate a centered layer of truncated octahedrons with proper space-filling tessellation.
    
    Creates a 3D grid of blocks with proper tessellation where each block has 14 neighbors
    (6 through square faces + 8 through hexagonal faces).
    
    Blocks have two types:
    - Type A: no rotation
    - Type B: rotated 90° around Z axis
    
    Args:
        nx: Number of blocks along X axis
        ny: Number of blocks along Y axis
        nz: Number of blocks along Z axis
        a: Block parameter a (half-size along X)
        b: Block parameter b (half-size along Y)
        c: Block parameter c (half-size along Z)
        alpha_deg: Bevel angle in degrees
        
    Returns:
        List of ConvexPolyhedron objects
    """
    blocks = []
    
    # Генерируем центры правильного замощения
    centers, indices = generate_tessellation_centers(nx, ny, nz, a, b, c)
    
    # Создаём блоки в каждом центре
    for idx, (block_center, (i, j, k, is_offset)) in enumerate(zip(centers, indices)):
        # Определяем тип блока по решающему правилу
        # Тип определяется по индексам ближайшего блока основной решётки (i, j, k)
        # Тип A: четная сумма индексов, Тип B: нечетная сумма
        # Блоки смещённой решётки имеют тот же тип, что и ближайший блок основной решётки
        is_type_b = (i + j + k) % 2 == 1
        
        # Создаём блок в нужном центре
        poly, _ = create_truncated_octahedron(
            a=a, b=b, c=c, alpha_deg=alpha_deg,
            center=block_center
        )
        
        # Если тип B, поворачиваем блок вокруг его центра
        if is_type_b:
            poly = rotate_polyhedron(
                poly,
                axis=np.array([0, 0, 1]),
                angle_deg=90.0,
                center=block_center
            )
        
        # Строим вершины и грани
        poly.build_vertices()
        poly.build_convex_hull()
        
        blocks.append(poly)
    
    return blocks
