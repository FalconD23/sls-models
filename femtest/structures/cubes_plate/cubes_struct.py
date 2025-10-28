



import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotation_matrix_body_diag_to_z():
    alpha = -math.pi/4
    Rz = np.array([
        [ math.cos(alpha), -math.sin(alpha), 0],
        [ math.sin(alpha),  math.cos(alpha), 0],
        [              0  ,               0 , 1],
    ])
    theta = -math.atan(math.sqrt(2))
    Ry = np.array([
        [               0, 1,              0],
        [ math.cos(theta), 0, math.sin(theta)],
        
        [-math.sin(theta), 0, math.cos(theta)],
    ])
    return Ry @ Rz

def make_cube_vertices(a):
    h = a / 2.0
    h *= (1.00 - 0.001)  #! face backlash <---------------------------------------
    return np.array([
        [-h, -h, -h],
        [ h, -h, -h],
        [ h,  h, -h],
        [-h,  h, -h],
        [-h, -h,  h],
        [ h, -h,  h],
        [ h,  h,  h],
        [-h,  h,  h],
    ])

# edges between cube vertices
CUBE_EDGES = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7),
]


def hex_grid_centers(L, W, s):
    """
    Возвращает список (x,y)-координат центров flat-top шестиугольников
    со стороной s, в сетке L столбцов и W строк.

    Каждый столбец смещён по вертикали на половину шага вниз для нечётных колонок.
    Горизонтальный шаг между центрами = 1.5 * s,
    вертикальный шаг между строками = sqrt(3) * s.
    """
    centers = []
    for row in range(W):
        for col in range(L):
            x = 1.5 * s * col
            y = math.sqrt(3) * s * (row + 0.5 * (col % 2))
            centers.append((x, y))
    return centers

def hexagon_vertices(cx, cy, s):
    angles = np.linspace(0, 2*math.pi, 7)[:-1]
    return [(cx + s*math.cos(a), cy + s*math.sin(a)) for a in angles]

def plot_structure(L=5, W=4, hex_side=1.0, cz=0.0):
    # prepare cube geometry
    R = rotation_matrix_body_diag_to_z()
    a = hex_side * math.sqrt(2)
    base_verts = make_cube_vertices(a) @ R.T
    
    # compute grid centers
    centers = hex_grid_centers(L, W, hex_side)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # plot hexagonal grid in the plane z=cz
    for (cx, cy) in centers:
        verts2d = hexagon_vertices(cx, cy, hex_side)
        xs, ys = zip(*(verts2d + [verts2d[0]]))
        zs = [cz]*len(xs)
        ax.plot(xs, ys, zs, color='green', linewidth=2.5)
    
    # plot each cube as wireframe
    for (cx, cy) in centers:
        verts3d = base_verts + np.array([cx, cy, cz])
        for i1, i2 in CUBE_EDGES:
            xline = [verts3d[i1,0], verts3d[i2,0]]
            yline = [verts3d[i1,1], verts3d[i2,1]]
            zline = [verts3d[i1,2], verts3d[i2,2]]
            ax.plot(xline, yline, zline, color='black', linewidth=4)
    
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# example
# plot_structure(L=1, W=3, hex_side=1.0, cz=0.2)




# --- Начало надстройки для STL ---

# список треугольников (по индексам вершин) для одной грани куба
_TRIANGLES = [
    (0,1,2), (0,2,3),   # низ
    (4,6,5), (4,7,6),   # верх
    (0,4,5), (0,5,1),   # перед
    (2,6,7), (2,7,3),   # зад
    (0,3,7), (0,7,4),   # лево
    (1,5,6), (1,6,2),   # право
]

def write_facets(f, verts):
    """
    Пишет в открытый файловый объект f все треугольники (_TRIANGLES)
    по списку вершин verts (8×3).
    """
    for (i1,i2,i3) in _TRIANGLES:
        v1, v2, v3 = verts[i1], verts[i2], verts[i3]
        # нормаль
        n = np.cross(v2-v1, v3-v1)
        n = n / np.linalg.norm(n)
        f.write(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("    outer loop\n")
        for v in (v1, v2, v3):
            f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("    endloop\n")
        f.write("  endfacet\n")

def generate_hex_grid_cubes_stl(L, W, hex_side, cz, filename):
    """
    То же, что plot_structure, но вместо отрисовки –
    пишет все кубы в один ASCII-STL.
    """
    R = rotation_matrix_body_diag_to_z()
    a = hex_side * math.sqrt(2)
    base_verts = make_cube_vertices(a) @ R.T
    centers = hex_grid_centers(L, W, hex_side)

    with open(filename, 'w') as f:
        f.write("solid hex_grid_cubes\n")
        for cx, cy in centers:
            verts = base_verts + np.array([cx, cy, cz])
            write_facets(f, verts)
        f.write("endsolid hex_grid_cubes\n")
    print(f"STL успешно записан в «{filename}» (L={L}, W={W}, hex_side={hex_side}, cz={cz})")

# парсер аргументов
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Без --stl: рисует в matplotlib, с --stl FILE: генерирует STL"
    )
    parser.add_argument("--L",       type=int,   default=8,   help="число столбцов")
    parser.add_argument("--W",       type=int,   default=8,   help="число строк")
    parser.add_argument("--hex_side",type=float, default=100.0, help="сторона шестиугольника")
    parser.add_argument("--cz",      type=float, default=0.0, help="высота центров кубов")
    parser.add_argument("--stl",     metavar="FILE",      help="имя выходного STL-файла")
    args = parser.parse_args()

    if args.stl:
        generate_hex_grid_cubes_stl(
            args.L, args.W, args.hex_side, args.cz, args.stl
        )
    else:
        plot_structure(
            L=args.L, W=args.W,
            hex_side=args.hex_side, cz=args.cz
        )
# --- Конец надстройки для STL ---
