
import FreeCAD, Part, ObjectsFem
from femtools import ccxtools
import re

# Hydra
import yaml
from pathlib import Path

 

def gen_list_from_txt(path, prefix='Face'):
    text = ''
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
    face_numbers = re.findall(rf'{prefix}(\d+)', text)
    return list(map(int, face_numbers))

# relative path doesn't work with freecad-python-cls running
config_path = Path("/home/ubnps23/tecHub/SLS_dev/sls-models/femtest/structures/bev_hex_prisms_plate/main_config.yaml")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)



ROOT_DIR = cfg["root_dir"]
def with_root(path):
    return f"{ROOT_DIR}/{path}"

STL_FILENAME = with_root(cfg["stl_filename"])
FACES_NUM = cfg["faces_num"]    
GUI = cfg["gui"]
TOLERANCE = cfg["tolerance"]

UNDER_PRESSURE_FILENAME = with_root(cfg["constraints"]["under_pressure_file"])

CONSTRAINT_FIXED_FILENAME = with_root(cfg["constraints"]["fixed_faces_file"])
CONSTRAINT_FIXED_FACES = gen_list_from_txt(CONSTRAINT_FIXED_FILENAME)

TOUCH_UNIT_NAME_PREFIX = cfg["constraints"]["TOUCH_UNIT_NAME_PREFIX"]
FACES_UNDER_PRESSURE = gen_list_from_txt(path=UNDER_PRESSURE_FILENAME, 
                                         prefix=TOUCH_UNIT_NAME_PREFIX)

FRICTION_ACTIVATE = cfg["contacts"]["activate_friction"]
CC_FILENAME = with_root(cfg["contacts"]["cc_pairs_file"])
CENTERS_DIST_FOR_CONTACT = cfg["contacts"]["centers_dist_for_contact"]
FRICTION_COEFF = cfg["contacts"]["friction_coeff"]
SLOPE_COEFF = cfg["contacts"]["slope_coeff"]

doc_prefix = cfg["doc_prefix"]
pressures_N_force = [i*1e7*1e0 / (10) for i in range(1, 3)]  # потому что прилагаются 3 силы к 3 вершинам
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
    solver_object.GeometricalNonlinearity = 'linear'
    solver_object.ThermoMechSteadyState = True
    solver_object.MatrixSolverType = 'default'
    solver_object.IterationsControlParameterTimeUse = False
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

    # 10. Фиксируем поверхности многогранников
    idx_faces_list = CONSTRAINT_FIXED_FACES
    fixed_faces_list = [f"Face{i}" for i in idx_faces_list]
    # Создаем объект фиксирующего ограничения для компаунда
    fixed_constraint = ObjectsFem.makeConstraintFixed(doc, "FemConstraintFixed")
    # Задаем ссылки на нужные грани компаунда
    fixed_constraint.References = [(cmp_obj, face) for face in fixed_faces_list] 
    analysis_object.addObject(fixed_constraint)


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


    # 12. Применяем внешние силы
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
            force_constraint.Force = base_force * (1 + (i / 10000))
            analysis_object.addObject(force_constraint)
        except Exception as e:
            print(f"Ошибка при создании силового ограничения для {touch_unit_name}: {e}")



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

    error = gmsh_mesh.create_mesh()
    if error:
        print("Ошибка создания сетки: ", error)
    else:
        print("Сетка успешно создана")
        print(f"Количество узлов: {len(femmesh_obj.FemMesh.Nodes)}")
        print(f"Количество элементов: {len(femmesh_obj.FemMesh.Volumes)}")

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