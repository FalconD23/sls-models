
import FreeCAD as App
import Part

LIN = 1e-5   # мм; 1e-6 на грани округления STEP, 1e-5 надёжнее
ANG = 1e-5   # рад

def canon(v):
    n = App.Vector(v)
    n.normalize()
    for c in (n.x, n.y, n.z):
        if abs(c) > 1e-9:
            return n if c > 0 else App.Vector(-n.x, -n.y, -n.z)
    return n

def same_plane(a, b):
    return a[0].getAngle(b[0]) < ANG and abs(a[1] - b[1]) < LIN

def groups(solid):
    items = []
    for f in solid.Faces:
        n = canon(f.Surface.Axis)
        items.append((f, (n, n.dot(f.CenterOfMass))))
    clusters = []
    for face, key in items:
        for cluster in clusters:
            if same_plane(cluster[0][1], key):
                cluster.append((face, key))
                break
        else:
            clusters.append([(face, key)])
    return clusters

def _ends(edge):
    return edge.Vertexes[0].Point, edge.Vertexes[-1].Point


def _same_edge(a, b, tol):
    """Одно и то же ребро: общий TShape или совпавшие концы.

    hashCode() в FreeCAD 0.21 — хеш, не идентификатор, на втором теле
    он склеивал разные рёбра и Part.Wire падал с 'command not done'.
    """
    if a.isSame(b):
        return True
    a1, a2 = _ends(a)
    b1, b2 = _ends(b)
    return (
        (a1.distanceToPoint(b1) <= tol and a2.distanceToPoint(b2) <= tol)
        or (a1.distanceToPoint(b2) <= tol and a2.distanceToPoint(b1) <= tol)
    )


def _boundary(faces, tol):
    records = []
    for face in faces:
        for edge in face.Edges:
            for rec in records:
                if _same_edge(rec[0], edge, tol):
                    rec[1] += 1
                    break
            else:
                records.append([edge, 1])
    return [edge for edge, count in records if count == 1]


def _chain(edges, tol):
    """Обход контура без Part.__sortEdges__.

    В FreeCAD 0.21 __sortEdges__ разворачивает ребро через новую кривую
    (BRepBuilderAPI_MakeEdge). На части рёбер это даёт
    'BRep_API: command not done'. reversed() меняет только ориентацию.
    """
    pool = list(edges)
    chain = [pool.pop(0)]
    start = chain[0].Vertexes[0].Point
    end = chain[0].Vertexes[-1].Point
    while pool:
        placed = False
        for i, edge in enumerate(pool):
            a = edge.Vertexes[0].Point
            b = edge.Vertexes[-1].Point
            if a.distanceToPoint(end) <= tol:
                chain.append(edge)
                end = b
            elif b.distanceToPoint(end) <= tol:
                chain.append(edge.reversed())
                end = a
            elif b.distanceToPoint(start) <= tol:
                chain.insert(0, edge)
                start = a
            elif a.distanceToPoint(start) <= tol:
                chain.insert(0, edge.reversed())
                start = b
            else:
                continue
            pool.pop(i)
            placed = True
            break
        if not placed:
            raise RuntimeError("контур грани не замкнут, осталось рёбер: %d" % len(pool))
    return chain


def one_face(faces):
    """Один многоугольник из треугольников одной плоскости.

    Контур — исходные рёбра, не новая полилиния: иначе соседние грани
    перестают стыковаться. Допуск ставится на рёбра до Part.Wire, потому
    что MakeWire соединяет вершины только в пределах их tolerance.
    """
    boundary = _boundary(faces, 1e-4)
    for edge in boundary:
        edge.fixTolerance(1e-4)
    wire = Part.Wire(_chain(boundary, 1e-4))
    plane = wire.findPlane(1e-4)
    if plane is None:
        return Part.Face(wire)
    return Part.Face(plane, wire)

def merge_solid(solid):
    faces = []
    for cluster in groups(solid):
        if len(cluster) == 1:
            faces.append(cluster[0][0])
        else:
            faces.append(one_face([face for face, _ in cluster]))
    # Part.Shell(список) в 0.21 после неудачной сшивки кидает
    # "Shape is not a shell". makeShell/makeSolid — штатный путь.
    shell = Part.makeShell(faces)
    return Part.makeSolid(shell)

shape = Part.Shape()
shape.read("plate_1370x1370x107.step")
solids = []
for index, solid in enumerate(shape.Solids, start=1):
    try:
        solids.append(merge_solid(solid))
    except Exception as exc:
        print("failed solid %d/%d, faces %d: %s" % (
            index, len(shape.Solids), len(solid.Faces), exc))
        raise
    if index == 1 or index % 50 == 0:
        print("merged %d/%d, faces %d -> %d" % (
            index, len(shape.Solids), len(solid.Faces), len(solids[-1].Faces)))

merged = Part.makeCompound(solids)
out_name = "plate_1370x1370x107_merged_LIN1e-5.step"
merged.exportStep(out_name)
print("wrote", out_name, "faces", sum(len(solid.Faces) for solid in solids))