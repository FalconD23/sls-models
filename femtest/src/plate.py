#!/usr/bin/env freecadcmd
import FreeCAD, Part, ObjectsFem
from femtools import ccxtools

# Отключаем GUI вообще
GUI = False

# Набор давлений (Н), которые хотим перебрать
patch_area_mm2 = 100*100
pressures_N = [i * 1e7 / (10 * patch_area_mm2) for i in range(1, 11)]  # 100, 200, …, 1000

results = []  # здесь будут кортежи (P, max_dz)

for curr_i, P in enumerate(pressures_N):

    # doc = App.newDocument(f"SteelPlate{curr_i}")
    doc = FreeCAD.newDocument(f"SteelPlate{curr_i}")
    print(f'\n\n===={curr_i}====\n\n')

    #todo ---------- 1. Геометрия пластины ----------
    Lx = Ly = 1370.0  # мм
    t  = 107.0         # толщина

    plate = doc.addObject("Part::Box", "Plate")
    plate.Length, plate.Width, plate.Height = Lx, Ly, t
    plate.Placement.Base = FreeCAD.Vector(0, 0, 0)

    #todo ---------- 2. «Площадка» 50×50 мм для давления ----------
    pad = doc.addObject("Part::Box", "LoadPatch")

    x0, y0 = Lx / 2, Ly / 2
    patch_size = 100.0       # размер заплатки (мм)
    patch_area_mm2 = patch_size * patch_size  # 2500

    # Перемещаем «заплатку» так, чтобы её центр был в (x0, y0)
    pad.Length = pad.Width = patch_size
    pad.Height = 0.1
    pad.Placement.Base = FreeCAD.Vector(
        x0 - patch_size/2,
        y0 - patch_size/2,
        t + 0.01  # пластина имеет толщину t, поэтому заплатка лежит на уровне z=t
    )
    doc.recompute()

    # делаем компаунд, чтобы сетка была общей
    compound = Part.Compound([plate.Shape, pad.Shape])
    cmp_obj  = doc.addObject("Part::Compound", "PlateCompound")
    cmp_obj.Links = [plate, pad]
    doc.recompute()

    #todo ---------- 3. Анализ и решатель ----------
    analysis = ObjectsFem.makeAnalysis(doc, "Analysis")
    solver   = ObjectsFem.makeSolverCalculixCcxTools(doc, "CalculiX")
    analysis.addObject(solver)

    #todo ---------- 4. Материал (сталь) ----------
    steel = ObjectsFem.makeMaterialSolid(doc, "Steel")
    m = steel.Material
    m["Name"]          = "Steel"
    m["YoungsModulus"] = "210000 MPa"
    m["PoissonRatio"]  = "0.30"
    m["Density"]       = "7850 kg/m^3"
    steel.Material = m
    analysis.addObject(steel)

    #todo ---------- 5. Граничные условия ----------
    # фиксируем ВСЕ боковые грани (Face2..Face5 для Part::Box)
    fix = ObjectsFem.makeConstraintFixed(doc, "FixSides")
    fix.References = [(plate, f"Face{i}") for i in (1,2,3,4)]
    analysis.addObject(fix)

    # ---- 5.5. Контакт между «заплаткой» и пластиной ----
    contact = ObjectsFem.makeConstraintContact(doc, "Contact_Patch_Plate")  
    contact.References = [
        (pad,   "Face5"),  # нижняя грань pad  :contentReference[oaicite:1]{index=1}
        (plate, "Face6")   # верхняя грань plate:contentReference[oaicite:2]{index=2}
    ]
    contact.Slope = 1e6
    contact.Friction = 0.3
    analysis.addObject(contact) 

    #todo ---------- 6. Давление 100 Н на 50×50 мм ----------
    # patch_area_mm2 = 50*50           # 2500 мм²
    # P = 2000.0 / patch_area_mm2       # Н/мм² = МПа·10⁻³
    press = ObjectsFem.makeConstraintPressure(doc, "CentralPressure")
    press.References = [(pad, "Face6")]        # верхняя грань заплатки
    # press.Pressure   = f"{P} N/mm^2"           # FreeCAD понимает такие ед-цы
    press.Pressure  = P          # FreeCAD понимает такие ед-цы
    analysis.addObject(press)

    #todo ---------- 7. Сетка Gmsh ----------
    mesh = ObjectsFem.makeMeshGmsh(doc, "Mesh")
    mesh.Part = cmp_obj
    doc.recompute()

    from femmesh.gmshtools import GmshTools as gt
    gm = gt(mesh)
    err = gm.create_mesh()
    if err:
        raise RuntimeError("Gmsh error: " + err)

    analysis.addObject(mesh)
    doc.recompute()

    #todo ---------- 8. Запуск CalculiX ----------
    if GUI:

        import FreeCADGui
        FreeCADGui.ActiveDocument.activeView().viewAxonometric()
        FreeCADGui.SendMsgToActiveView("ViewFit")

        import FemGui
        FemGui.setActiveAnalysis(analysis)


    fea = ccxtools.FemToolsCcx()
    fea.purge_results()
    fea.run()

    #todo ---------- 9. Пост-процесс ----------
    res = next(o for o in analysis.Group if o.isDerivedFrom("Fem::FemResultObject"))
    dz = [v[2] for v in res.DisplacementVectors]
    max_dz = max(dz, key=abs)
    # print("Max |dz| = %.6f mm" % max_dz)

    # показать деформацию, если есть GUI
    if GUI:
        mesh.ViewObject.setNodeDisplacementByVectors(res.NodeNumbers, res.DisplacementVectors)
        mesh.ViewObject.applyDisplacement(20)
        # print("В GUI отображена деформированная форма x20")



    results.append((P, max_dz))

    # уничтожаем документ перед следующим циклом
    if not GUI:
        FreeCAD.closeDocument(doc.Name)

# Печатаем два списка: давлений и соответствующих max_dz
press_list = [p for p, d in results]
dz_list    = [d for p, d in results]
print("Pressures (N):", press_list)
print("Max dz (mm):", dz_list)
