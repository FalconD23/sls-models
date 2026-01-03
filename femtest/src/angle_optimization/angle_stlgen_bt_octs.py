"""
Main entry point for STL layer generation.

This module provides a clean interface for generating STL models
of beveled truncated octahedrons layers, replacing the legacy procedural code.
"""

import numpy as np
import sys
import os
from pathlib import Path

script_path = Path(__file__).absolute()
script_dir = script_path.parent
current_work_dir = os.getcwd()

# Добавляем путь к stlgen (родительская директория) для импорта export, geometry и т.д.
# Это должно быть сделано ПЕРЕД импортом bt_octs_utils, так как bt_octs_utils тоже использует эти модули
stlgen_root = script_path.parent.parent.parent.parent / "stlgen"
if not stlgen_root.exists():
    raise ImportError(f"STLgen root directory not found: {stlgen_root}")
# Добавляем в начало, чтобы иметь приоритет
if str(stlgen_root) not in sys.path:
    sys.path.insert(0, str(stlgen_root))

# stlgen_dir = script_path.parent.parent.parent.parent / "stlgen" / "tests"
# if not stlgen_dir.exists():
#     raise ImportError(f"STLgen directory not found: {stlgen_dir}")
# sys.path.insert(0, str(stlgen_dir))

# Добавляем путь к stlgen/tests для импорта bt_octs_utils
stlgen_tests_dir = stlgen_root / "tests"
if not stlgen_tests_dir.exists():
    raise ImportError(f"STLgen tests directory not found: {stlgen_tests_dir}")
# Добавляем после stlgen_root, чтобы bt_octs_utils мог найти модули из stlgen
if str(stlgen_tests_dir) not in sys.path:
    sys.path.insert(0, str(stlgen_tests_dir))

# Импортируем модули (stlgen_root должен быть в пути для работы этих импортов)
from export.stl_exporter import STLExporter
# bt_octs_utils сам добавит путь к stlgen_root, но он уже должен быть там
from bt_octs_utils import LayerGen



def generate_stl_for_angle(angle_degrees=37, output_dir=None):
    # Параметры слоя
    nx, ny, nz = 10, 10, 1
    scale = 1300/160
    a, b, c = 8.0*scale, 8.0*scale, 6.0*scale
    alpha_deg = angle_degrees
    
    print(f"\nПараметры слоя:")
    print(f"  Размеры: {nx}×{ny}×{nz} блоков")
    print(f"  Параметры блока: a={a}, b={b}, c={c}, α={alpha_deg}°")
    
    # Создаём слой
    print(f"\nСоздание слоя из {nx * ny * nz} блоков...")
    blocks = LayerGen(nx=nx, ny=ny, nz=nz, a=a, b=b, c=c, alpha_deg=alpha_deg)
    
    print(f"✓ Создано блоков: {len(blocks)}")
    print(f"  (основная решётка: {nx * ny * nz}, смещённая решётка: {nx * ny * nz})")
    
    # Подсчитываем типы блоков
    # Каждый блок основной решётки имеет соответствующий блок смещённой решётки с тем же типом
    type_a_count = 0
    type_b_count = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Каждый блок основной решётки даёт 2 блока (основной + смещённый) с одинаковым типом
                if (i + j + k) % 2 == 0:
                    type_a_count += 2  # Основной + смещённый
                else:
                    type_b_count += 2  # Основной + смещённый
    
    # Экспорт всех блоков в один STL файл
    print("\nЭкспорт слоя в STL...")
    
    
    

    
    exporter = STLExporter(tolerance=1e-5)
    
    # Собираем все грани всех блоков
    all_blocks = []
    for poly in blocks:
        if poly.faces:
            all_blocks.append(poly.faces)
    
    if not all_blocks:
        print("⚠️ Нет граней для экспорта")
        return
    
    # Формируем имя файла
    stl_filename = f"layer_gen_{nx}x{ny}x{nz}_a{a}_b{b}_c{c}_alpha{alpha_deg}.stl"
    output_file = Path(__file__).parent / stl_filename
    solid_name = f"layer_gen_{nx}x{ny}x{nz}_a{a}_b{b}_c{c}_alpha{alpha_deg}"
    
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    try:
        # Экспортируем
        exporter.write_stl(
            blocks=all_blocks,
            filename=str(output_file),
            solid_name=solid_name,
            format="ascii"
        )
        print(f"✓ STL файл создан: {stl_filename}")
        print(f"✓ Размер файла: {output_file.stat().st_size} байт")

        stl_path = os.path.join(output_dir, stl_filename)
        print(f"Generated STL for angle {angle_degrees:.1f}°: {stl_path}")
        
        return stl_path
    finally:
        os.chdir(original_cwd)


def main():
    """
    Main function demonstrating the new OOP interface.
    
    This replaces the legacy main.py with a clean, object-oriented approach.
    """
    # Generate STL for angle 0 degrees as example
    stl_path = generate_stl_for_angle(
        angle_degrees=0,
        filename_prefix="run_angle_bev_hex_prisms"
    )
    print(f"STL file generated: {stl_path}")



if __name__ == "__main__":
    main()
