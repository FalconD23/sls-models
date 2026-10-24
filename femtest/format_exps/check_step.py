"""Проверка многотельного STEP после склейки граней.

Запуск из каталога с файлом, тем же интерпретатором, что и stl2step.py:

    cd /workspace/femtest/format_exps
    freecadcmd check_step.py
"""

import Part

STEP_NAME = "plate_1370x1370x107_merged_LIN1e-5.step"
EXPECTED_SOLIDS = 267
EXPECTED_FACES_PER_SOLID = 6
OVERLAP_VOLUME = 1e-3  # мм^3, общий объём выше этого — врезка, не касание

shape = Part.Shape()
shape.read(STEP_NAME)
box = shape.BoundBox

print("solids", len(shape.Solids))
print("faces", len(shape.Faces))
print(
    "bbox",
    round(box.XLength, 2),
    round(box.YLength, 2),
    round(box.ZLength, 2),
)
print("volume", round(shape.Volume, 1))

bad = []
for index, solid in enumerate(shape.Solids):
    print("solid", index, "volume", round(solid.Volume, 3))
    if (
        (not solid.isValid())
        or (not solid.isClosed())
        or len(solid.Faces) != EXPECTED_FACES_PER_SOLID
        or solid.Volume <= 0
    ):
        bad.append(
            (
                index,
                solid.isValid(),
                solid.isClosed(),
                len(solid.Faces),
                solid.Volume,
            )
        )
print("bad solids", len(bad))
for row in bad[:20]:
    print(row)

hits = []
solids = shape.Solids
for i in range(len(solids)):
    box_a = solids[i].BoundBox
    for j in range(i + 1, len(solids)):
        box_b = solids[j].BoundBox
        if box_a.XMax < box_b.XMin or box_b.XMax < box_a.XMin:
            continue
        if box_a.YMax < box_b.YMin or box_b.YMax < box_a.YMin:
            continue
        if box_a.ZMax < box_b.ZMin or box_b.ZMax < box_a.ZMin:
            continue
        common = solids[i].common(solids[j])
        if common.Volume > OVERLAP_VOLUME:
            hits.append((i, j, common.Volume))
print("overlaps", len(hits))
for row in hits[:20]:
    print(row)

if len(shape.Solids) != EXPECTED_SOLIDS or bad or hits:
    raise SystemExit(1)
print("ok")
