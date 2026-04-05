
# from CAD_testing.cad_analysis.planes_cmp.tetrahedrons_sls.fem_eval_plus_Hydra import STL_FILENAME
import FreeCAD, Part, ObjectsFem
from femtools import ccxtools
import re
import subprocess

# Hydra
import yaml
from pathlib import Path
                                                            


def gen_list_from_txt(path, prefix='Face'):
    text = ''
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
    face_numbers = re.findall(rf'{prefix}(\d+)', text)
    return list(map(int, face_numbers))

# Path resolved relative to this script file — works locally and inside Docker container
# tetrahedrons_plate, cubes_plate, bev_hex_prisms_plate, bev_trunc_octahedrons
_SCRIPT_DIR = Path(__file__).resolve().parent
config_path = _SCRIPT_DIR / "structures" / "bev_trunc_octahedrons" / "main_config.yaml"
# config_path = "/home/ubnps23/tecHub/SLS_dev/sls-models/femtest/structures/bev_trunc_octahedrons/main_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)



# If root_dir is __AUTO__ (portable mode), compute it relative to this script
_AUTO_ROOT = str(_SCRIPT_DIR / "structures" / "bev_trunc_octahedrons")
ROOT_DIR = _AUTO_ROOT if cfg.get("root_dir") == "__AUTO__" else cfg["root_dir"]
def with_root(path):
    return f"{ROOT_DIR}/{path}"

for angle in range(37, 38, 1):
# for angle in [23]:
    STL_FILENAME = with_root(cfg["stl_filename"])
    # STL_FILENAME = with_root(f'RUN_layer_BTOCTS_alpha{angle}_10x10x1_a65.0_b65.0_c48.75.stl')
    # STL_FILENAME = with_root(f'EXTRA_RUN_layer_BTOCTS_alpha{angle}_20x20x1_a65.0_b65.0_c48.75.stl')
    FACES_NUM = cfg["faces_num"]    
    GUI = cfg["gui"]
    TOLERANCE = cfg["tolerance"]

    UNDER_PRESSURE_FILENAME = with_root(cfg["constraints"]["under_pressure_file"])
    LIST_MODE = False

    CONSTRAINT_FIXED_FILENAME = with_root(cfg["constraints"]["fixed_faces_file"])
    CONSTRAINT_FIXED_FACES = gen_list_from_txt(CONSTRAINT_FIXED_FILENAME)

    TOUCH_UNIT_NAME_PREFIX = cfg["constraints"]["TOUCH_UNIT_NAME_PREFIX"]
    BORDER_TOLERANCE = cfg["constraints"]["border_tolerance"]
    FACES_UNDER_PRESSURE = gen_list_from_txt(path=UNDER_PRESSURE_FILENAME, 
                                            prefix=TOUCH_UNIT_NAME_PREFIX)
    APPLY_TOLERANCE = cfg["constraints"]["apply_tolerance"]

    FRICTION_ACTIVATE = cfg["contacts"]["activate_friction"]
    CC_FILENAME = with_root(cfg["contacts"]["cc_pairs_file"])
    CENTERS_DIST_FOR_CONTACT = cfg["contacts"]["centers_dist_for_contact"]
    FRICTION_COEFF = cfg["contacts"]["friction_coeff"]
    SLOPE_COEFF = cfg["contacts"]["slope_coeff"]

    doc_prefix = cfg["doc_prefix"] + f'_angle{angle}_'
    pressures_N_force = [i*1e7*1e0 / (10) for i in range(10, 11)]  # потому что прилагаются 3 силы к 3 вершинам
    results = []

    # doc = App.newDocument("Imported_3D_Model_FEM")

    for curr_i, force in enumerate(pressures_N_force, start=1):
        # 1. Создаем новый документ
        # doc = App.newDocument("Imported_3D_Model_FEM")
        doc = FreeCAD.newDocument(f"{doc_prefix}{curr_i}")
        print(f'\n\n===={curr_i}====\n\n')


        # 2. Импортируем STL модель как Mesh
        import Mesh
        stl_file_path = STL_FILENAME

        mesh_obj = Mesh.Mesh(stl_file_path)
        print("STL файл импортирован как Mesh объект")

        # 3. Преобразуем Mesh в Shape
        import Part
        shape = Part.Shape()
        tolerance = TOLERANCE  # Допуск аппроксимации (подберите опытным путем)
        shape.makeShapeFromMesh(mesh_obj.Topology, tolerance)

        # 4. Разбиваем Shape на отдельные полигональные тела
        # Предполагается, что каждый многогранник образован FACES_NUM последовательными гранями (индексы 0..19, FACES_NUM..39, ...)
        num_faces = len(shape.Faces)
        poly_count = num_faces // FACES_NUM
        solids_list = []  # Список для хранения созданных твердых тел

        # poly_idx = [i for i in range(poly_count) if i not in [117 // 20]]

        # for i in poly_idx:
        for i in range(poly_count):
            start = i * FACES_NUM
            end = start + FACES_NUM
            faces_group = shape.Faces[start:end]
            try:
                # Создаем оболочку из группы граней
                shell = Part.Shell(faces_group) 
                # Преобразуем оболочку в твердое тело
                solid_poly = Part.makeSolid(shell)
                solid_name = f"SolidModel_{i+1}"
                solid_obj = doc.addObject("Part::Feature", solid_name)
                solid_obj.Shape = solid_poly
                solids_list.append(solid_obj)
            except Exception as e:
                print(f"Ошибка при создании твердого тела для многогранника {i+1}: {e}")

        doc.recompute()

        # 5. делаем компаунд, чтобы сетка была общей
        # shapes = [obj.Shape for obj in solids_list]
        compound = Part.Compound([obj.Shape for obj in solids_list])
        cmp_obj  = doc.addObject("Part::Compound", "PlateCompound")
        cmp_obj.Links = solids_list
        doc.recompute()



        # 6. Настраиваем визуализацию модели


        if GUI:
            import FemGui
            import FreeCADGui
            FreeCADGui.ActiveDocument.activeView().viewAxonometric()
            FreeCADGui.SendMsgToActiveView("ViewFit")

        # 7. Создаем объект анализа FEM
        import ObjectsFem
        analysis_object = ObjectsFem.makeAnalysis(doc, "Analysis")

        # 8. Создаем решатель CalculiX для FEM анализа
        solver_object = ObjectsFem.makeSolverCalculixCcxTools(doc, "CalculiX")
        solver_object.GeometricalNonlinearity = 'nonlinear' #'linear'
        solver_object.ThermoMechSteadyState = True
        solver_object.MatrixSolverType = 'default'
        solver_object.IterationsControlParameterTimeUse = True #False
        analysis_object.addObject(solver_object)

        # 9. Определяем материал (сталь) для анализа
        material_object = ObjectsFem.makeMaterialSolid(doc, "SolidMaterial")
        mat = material_object.Material
        mat['Name'] = "Steel-Generic"
        mat['YoungsModulus'] = "210000 MPa"
        mat['PoissonRatio'] = "0.30"
        mat['Density'] = "7850 kg/m^3" 
        material_object.Material = mat
        analysis_object.addObject(material_object) 

        # 10. Фиксируем внешние грани блоков (относительно центра конструкции)

        #* legacy version with list of faces to fix
        # idx_faces_list = CONSTRAINT_FIXED_FACES
        # fixed_faces_list = [f"Face{i}" for i in idx_faces_list]
        # # Создаем объект фиксирующего ограничения для компаунда
        # fixed_constraint = ObjectsFem.makeConstraintFixed(doc, "FemConstraintFixed")
        # # Задаем ссылки на нужные грани компаунда
        # fixed_constraint.References = [(cmp_obj, face) for face in fixed_faces_list] 
        # analysis_object.addObject(fixed_constraint)

        #* new version with tolerance and centroid relative
        # Найдём центры всех блоков
        centers = []
        for solid in solids_list:
            com = solid.Shape.CenterOfMass
            centers.append((solid, com))

        if not centers:
            print("Нет блоков для анализа центра структуры")
        else:
            # Вычисляем центр всей структуры (средний центр всех центров)
            avg_x = sum(c[1].x for c in centers) / len(centers)
            avg_y = sum(c[1].y for c in centers) / len(centers)
            avg_z = sum(c[1].z for c in centers) / len(centers)

            center_struct = FreeCAD.Vector(avg_x, avg_y, avg_z)
            print(f"Center of structure at ({avg_x:.3f}, {avg_y:.3f}, {avg_z:.3f})")

            # Находим максимальные отклонения от центра структуры
            max_dx = max(abs(c[1].x - avg_x) for c in centers)
            max_dy = max(abs(c[1].y - avg_y) for c in centers)
            max_dz = max(abs(c[1].z - avg_z) for c in centers)

            # порог (0.95)
            border_tolerance = BORDER_TOLERANCE
            thr_dx = max_dx * border_tolerance
            thr_dy = max_dy * border_tolerance
            thr_dz = max_dz * border_tolerance * 20

            print(f"Max deviations: dX={max_dx:.3f}, dY={max_dy:.3f}, dZ={max_dz:.3f}")
            print(f"Thresholds dX>={thr_dx:.3f}, dY>={thr_dy:.3f}, dZ>={thr_dz:.3f}")

            # Собираем список граней для фиксации
            faces_to_fix = []

            for solid, com in centers:
                dx = abs(com.x - avg_x)
                dy = abs(com.y - avg_y)
                dz = abs(com.z - avg_z)

                # если центр блока близок к границе по любой из осей
                if dx >= thr_dx or dy >= thr_dy or dz >= thr_dz:
                    for idx, face in enumerate(solid.Shape.Faces, start=1):
                        face_name = f"Face{idx}"
                        faces_to_fix.append((solid, face_name))
                    print(f"Block {solid.Name} at ({com.x:.3f},{com.y:.3f},{com.z:.3f}) FIXED")

            # Создаём constraint fixed
            fixed_constraint = ObjectsFem.makeConstraintFixed(doc, "FemConstraintFixed")
            fixed_constraint.References = faces_to_fix
            analysis_object.addObject(fixed_constraint)

            print(f"Total fixed faces: {len(faces_to_fix)}")


        #? 11. Автоматическое задание контактных ограничений с трением между соседними блоками
        import os
        import csv

        def face_center(face):
            center = face.CenterOfMass
            return center

        cc_pairs = []
        # filename = 'cc_pairs.csv'
        filename = CC_FILENAME
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            print("Файл со списком контактов существует и не пуст")
        else:
            # Для каждого блока и его граней находим ближайшую грань другого блока
            for i in range(len(solids_list)):
                for j in range(len(solids_list)):
                    if i == j: 
                        continue    
                    A = solids_list[i]
                    B = solids_list[j]
                    # назовём best — ближайшая
                    for idxA, faceA in enumerate(A.Shape.Faces, start=1):
                        for idxB, faceB in enumerate(B.Shape.Faces, start=1):
                            centerA = face_center(faceA)
                            centerB = face_center(faceB)
                            if (centerA - centerB).Length < CENTERS_DIST_FOR_CONTACT:
                                cc_pairs.append((i, idxA, j, idxB))

            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(cc_pairs)  # Записываем все кортежи


        if FRICTION_ACTIVATE:
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                cc_pairs = [tuple(map(int, row)) for row in reader]  # Преобразуем строки в числа


            for i, idxA, j, idxB in cc_pairs:
                A = solids_list[i]
                B = solids_list[j]
                cname = f"Contact_{A.Name}_F{idxA}_{B.Name}_F{idxB}"
                # print(cname)
                cc = ObjectsFem.makeConstraintContact(doc, cname)
                cc.References = [(A, f"Face{idxA}"), (B, f"Face{idxB}")]
                cc.Friction = FRICTION_COEFF
                cc.Slope = SLOPE_COEFF  #!<----------------
                analysis_object.addObject(cc)

        # 12-LIST VERSION. Применяем внешние силы
        if LIST_MODE:
        # Здесь направление силы вычисляется функцией, а величина силы масштабируется в зависимости от номера грани.
            def calculate_direction(face_number): # Пример логики: сила направлена вдоль оси -Z, но величина может зависеть от номера
                return FreeCAD.Vector(0, 0, -1)  # [FreeCAD.Vector: https://wiki.freecad.org/FreeCAD_Vector]

            base_force = force / len(FACES_UNDER_PRESSURE) # базовая величина силы

            faces_under_pressure_idx = FACES_UNDER_PRESSURE
            # faces_under_pressure_idx = [443, 421, 325]  # faces
            # faces_under_pressure_idx = list(range(100, 131))
            for i in faces_under_pressure_idx:
                touch_unit_name = f"{TOUCH_UNIT_NAME_PREFIX}{i}"
                # touch_unit_name = f"Face{i}"
                try:
                    force_constraint = ObjectsFem.makeConstraintForce(doc, f"Force_{touch_unit_name}")
                    force_constraint.References = [(cmp_obj, touch_unit_name)]
                    # Вычисляем направление силы
                    direction_vector = calculate_direction(i)
                    # Для задания направления создаем вспомогательное ребро
                    direction_edge = Part.makeLine(FreeCAD.Vector(0, 0, 0), direction_vector)
                    direction_obj = doc.addObject("Part::Feature", f"Direction_{touch_unit_name}")
                    direction_obj.Shape = direction_edge
                    force_constraint.Direction = (direction_obj, ["Edge1"])
                    
                    # Величина силы пропорциональна (i - 99)
                    force_constraint.Force = base_force * (1 + 0)
                    analysis_object.addObject(force_constraint)
                except Exception as e:
                    print(f"Ошибка при создании силового ограничения для {touch_unit_name}: {e}")

        else:
            # 12. Применяем внешние силы к горизонтальным граням блоков в +/-5% от центра
            # 1) вычисляем центроид блоков (если не было ранее)
            centers = [(solid, solid.Shape.CenterOfMass) for solid in solids_list]
            if not centers:
                print("No solids for force application")
            else:
                avg_x = sum(c[1].x for c in centers) / len(centers)
                avg_y = sum(c[1].y for c in centers) / len(centers)
                avg_z = sum(c[1].z for c in centers) / len(centers)

                max_dx = max(abs(c[1].x - avg_x) for c in centers)
                max_dy = max(abs(c[1].y - avg_y) for c in centers)
                max_dz = max(abs(c[1].z - avg_z) for c in centers)

                # коэффициент 20%
                apply_tolerance = APPLY_TOLERANCE
                lim_dx = max_dx * apply_tolerance
                lim_dy = max_dy * apply_tolerance

                print(f"Force region limits: dx<={lim_dx:.3f}, dy<={lim_dy:.3f}")

                # собираем все горизонтальные грани и их Z
                faces_all = []  # (solid, idx, face_z)

                for solid, com in centers:
                    for idx, face in enumerate(solid.Shape.Faces, start=1):
                        z = face.CenterOfMass.z
                        faces_all.append((solid, idx, z))
                # глобальный максимум Z (верх конструкции)
                global_max_face_z = max(z for _, _, z in faces_all)
                print(f"Global top face Z = {global_max_face_z:.6f}")

                # 2) Проверка: грань горизонтальная И выше центра блока (верхняя грань)
                def is_top_horizontal_face(face, solid_com, tol_angle=0.01, tol_z=1e-4):
                    # нормаль грани
                    normal = face.normalAt(0.5, 0.5)

                    # 1) горизонтальность: нормаль почти по ±Z
                    if abs(abs(normal.z) - 1.0) >= tol_angle:
                        return False

                    # 2) грань выше центра блока
                    face_com = face.CenterOfMass
                    if (face_com.z <= solid_com.z + tol_z) or (face_com.z < global_max_face_z - tol_z): #!<---------------- avg_z*1.1 is a hack to avoid the case when the face is very close to the center of the block
                        return False

                    return True


                # 3) собираем список граней, к которым надо приложить силы
                horizontal_faces = []  # (solid, face_index)

                for solid, com in centers:
                    dx = abs(com.x - avg_x)
                    dy = abs(com.y - avg_y)

                    # проверка попадания центра блока в 5% площадь
                    if dx <= lim_dx and dy <= lim_dy:
                        for idx, face in enumerate(solid.Shape.Faces, start=1):
                            if is_top_horizontal_face(face, com):
                                horizontal_faces.append((solid, idx))
                                print(f"Candidate top-horizontal face for force: {solid.Name}-Face{idx}")

                if not horizontal_faces:
                    print("No top-horizontal faces found in central region")
                else:
                    # 4) равномерное давление: сначала считаем суммарную площадь
                    total_area = 0.0
                    for solid, idx in horizontal_faces:
                        face = solid.Shape.Faces[idx - 1]
                        total_area += face.Area

                    if total_area <= 0:
                        raise RuntimeError("Total area for pressure application is zero")

                    # давление = сила / площадь
                    pressure_value = force / total_area  # [N / mm^2] если модель в мм
                    print(f"Total area = {total_area:.6f}, pressure = {pressure_value:.6e}")
                    for solid, idx in horizontal_faces:
                        face_name = f"Face{idx}"
                        cname = f"Pressure_{solid.Name}_F{idx}"

                        try:
                            pc = ObjectsFem.makeConstraintPressure(doc, cname)
                            pc.References = [(solid, face_name)]

                            # давление одно и то же для всех граней
                            pc.Pressure = pressure_value
                            pc.Scale = 1

                            # направление: по нормали грани (по умолчанию)
                            pc.Reversed = False  # если окажется, что давит "вверх" — поставить True

                            analysis_object.addObject(pc)

                        except Exception as e:
                            print(f"Error applying pressure to {solid.Name}-Face{idx}: {e}")


        # 13. Создаем FEM-сетку с использованием Gmsh на компаунде
        femmesh_obj = ObjectsFem.makeMeshGmsh(doc, "CompoundMesh")
        femmesh_obj.Part = cmp_obj
        doc.recompute()

        from femmesh.gmshtools import GmshTools as gt
        gmsh_mesh = gt(femmesh_obj)

        # Проверяем геометрию перед созданием сетки
        print(f"Количество твердых тел в компаунде: {len(cmp_obj.Shape.Solids)}")
        # for i, solid in enumerate(cmp_obj.Shape.Solids):
        #     print(f"[DEBUG]: Твердое тело {i+1}: {len(solid.Faces)} граней, объем = {solid.Volume:.6f}")

        try:
            error = gmsh_mesh.create_mesh()
        except Exception as exc:
            msg = str(exc)
            print(f"Gmsh raised exception: {msg}")

            # Try to extract /tmp/fcfem_* directory from exception and provide
            # extra diagnostics to understand why .unv was not created.
            m = re.search(r"(/tmp/fcfem_[^/\s]+)", msg)
            if m:
                tmp_dir = Path(m.group(1))
                geo_file = tmp_dir / "shape2mesh.geo"
                unv_file = tmp_dir / "PlateCompound_Mesh.unv"
                print(f"Gmsh temp dir: {tmp_dir}")
                if tmp_dir.exists():
                    print("Temp dir files:")
                    for p in sorted(tmp_dir.iterdir()):
                        print(f"  - {p.name}")
                else:
                    print("Temp dir does not exist.")

                if geo_file.exists():
                    print("Running manual gmsh diagnostics...")
                    cmd = [
                        "gmsh",
                        str(geo_file),
                        "-3",
                        "-format", "unv",
                        "-o", str(unv_file),
                        "-v", "4",
                    ]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    print(f"gmsh exit code: {res.returncode}")
                    if res.stdout:
                        print("gmsh stdout tail:")
                        print("\n".join(res.stdout.splitlines()[-20:]))
                    if res.stderr:
                        print("gmsh stderr tail:")
                        print("\n".join(res.stderr.splitlines()[-20:]))

            raise RuntimeError(f"Gmsh mesh creation crashed: {msg}") from exc

        if error:
            raise RuntimeError(f"Gmsh mesh creation failed: {error}")

        n_nodes = len(femmesh_obj.FemMesh.Nodes)
        n_volumes = len(femmesh_obj.FemMesh.Volumes)
        n_faces = len(getattr(femmesh_obj.FemMesh, "Faces", {}))
        n_edges = len(getattr(femmesh_obj.FemMesh, "Edges", {}))

        # Fail fast with explicit diagnostics instead of failing later in fea.run()
        if n_volumes == 0 and n_faces == 0 and n_edges == 0:
            raise RuntimeError(
                "Gmsh returned empty mesh (0 volumes, 0 faces, 0 edges). "
                "Check geometry validity, meshing parameters and gmsh output files in /tmp/fcfem_*."
            )

        print("Сетка успешно создана")
        print(f"Количество узлов: {n_nodes}")
        print(
            "Элементы сетки: "
            f"volumes={n_volumes}, faces={n_faces}, edges={n_edges}"
        )

        analysis_object.addObject(femmesh_obj)  # [addObject: https://wiki.freecad.org/FEM_Workbench]
        doc.recompute()



        # 14. Устанавливаем активный анализ для визуализации результатов
        if GUI:
            FemGui.setActiveAnalysis(analysis_object)

        # 15. Запускаем анализ "все в одном"
        from femtools import ccxtools  
        fea = ccxtools.FemToolsCcx()
        fea.purge_results()  
        fea.run() 

        #todo ---------- 9. Пост-процесс ----------
        res = next(o for o in analysis_object.Group if o.isDerivedFrom("Fem::FemResultObject"))
        dz = [v[2] for v in res.DisplacementVectors]
        max_dz = max(dz, key=abs)
        # print("Max |dz| = %.6f mm" % max_dz)

        # показать деформацию, если есть GUI
        # if GUI:
        #     femmesh_obj.ViewObject.setNodeDisplacementByVectors(res.NodeNumbers, res.DisplacementVectors)
        #     femmesh_obj.ViewObject.applyDisplacement(20)
            # print("В GUI отображена деформированная форма x20")

        results.append((force, max_dz))
        # уничтожаем документ перед следующим циклом
        if not GUI:
            FreeCAD.closeDocument(doc.Name)

    # Печатаем два списка: давлений и соответствующих max_dz
    press_list = [p for p, _ in results]
    dz_list    = [d for _, d in results]
    print("Pressures (N):", press_list)
    print("Max dz (mm):", dz_list)