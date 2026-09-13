"""STEP export stub — implement later (FreeCAD / pythonOCC / cascade)."""

from __future__ import annotations

from typing import Any, Sequence


def export_step(
    blocks: Sequence[Any],
    filename: str,
    **kwargs: Any,
) -> None:
    """
    Заглушка генератора STEP.

    Позже: собрать solid из ConvexPolyhedron (грани/плоскости) и записать .step.
    Переключение как альтернатива STL — флаг --format step в generate.py.
    """
    raise NotImplementedError(
        "STEP export is a stub. Use --format stl for now. "
        f"Requested file: {filename}, blocks={len(blocks)}, extra={kwargs}"
    )
