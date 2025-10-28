# Генератор STL Слоев

Современный объектно-ориентированный Python пакет для генерации STL моделей слоев скошенных шестигранных призм. Этот пакет заменяет устаревший процедурный код чистой, поддерживаемой архитектурой.

## Возможности

- **Объектно-Ориентированный Дизайн**: Четкое разделение ответственности с выделенными классами для каждого компонента
- **Надежная Геометрия**: Правильная валидация и обработка ошибок для геометрических операций
- **Гибкие Трансформации**: Поддержка преобразований систем координат и цилиндрического изгиба
- **Множественные Форматы Экспорта**: Экспорт как в ASCII, так и в бинарный STL
- **Комплексное Тестирование**: Включены модульные и интеграционные тесты
- **Типобезопасность**: Полные аннотации типов для лучшей поддержки IDE и ясности кода

## Архитектура

### Основные Компоненты

- **GeometryConfig**: Инкапсулирует все геометрические параметры (радиус, высота, угол скоса и т.д.)
- **CoordinateSystem**: Обрабатывает преобразования координат между системами отсчета
- **BeveledPrism**: Представляет одну скошенную шестигранную призму с 8 вершинами
- **PrismLayer**: Управляет шестигранной сеткой скошенных призм с чередующимися ориентациями
- **ConvexHullSolver**: Вычисляет точки пересечения и строит выпуклые оболочки для генерации граней
- **CylindricalTransform**: Применяет цилиндрические трансформации изгиба
- **STLExporter**: Обрабатывает экспорт как в ASCII, так и в бинарный STL форматы
- **LayerGenerator**: Главный фасадный класс, который оркестрирует весь процесс

### Ключевые Улучшения по сравнению с Устаревшим Кодом

1. **Устранено Дублирование Кода**: Объединены файлы `generate_unit_block*.py` в единый класс `BeveledPrism`
2. **Удалены Глобальные Переменные**: Все параметры инкапсулированы в `GeometryConfig`
3. **Улучшена Обработка Ошибок**: Заменен `try/except pass` на правильную валидацию
4. **Типобезопасность**: Полные аннотации типов во всем коде
5. **Модульный Дизайн**: Каждый компонент может использоваться независимо
6. **Комплексное Тестирование**: Включены модульные и интеграционные тесты

## Установка

```bash
# Установка в режиме разработки
cd /path/to/SLS_dev
pip install -e models/stlgen
```

## Быстрый Старт

```python
import numpy as np
from models.stlgen import GeometryConfig, LayerGenerator

# Создание геометрической конфигурации
config = GeometryConfig(
    radius=4.0,
    height=1.0,
    bev_angle=np.radians(30),
    size_trick=1.1
)

# Создание генератора слоев
generator = LayerGenerator(config, bend_radius=30e6)

# Генерация полного слоя со всеми этапами обработки
layer = generator.generate_complete_layer(
    x_num=11,
    y_num=11,
    output_filename="bf8_sls_ascii.stl",
    solid_name="bf8_sls",
    format="ascii"
)

print(f"Сгенерирован слой с {len(layer)} призмами")
```

## Продвинутое Использование

### Пошаговая Обработка

```python
# Генерация слоя
generator = LayerGenerator(config)
layer = generator.generate_layer(x_num=11, y_num=11)

# Применение цилиндрического изгиба
bent_layer = generator.apply_bending(bend_radius=30e6)

# Вычисление треугольных граней
faces = generator.compute_faces()

# Экспорт в STL
generator.export_stl("output.stl", format="binary")
```

### Пользовательские Системы Координат

```python
from models.stlgen import CoordinateSystem

# Создание пользовательской системы координат
custom_cs = CoordinateSystem.from_vectors(
    i=[1, 0, 0],
    j=[0, 1, 0],
    k=[0, 0, 1],
    origin=[5, 0, 0]
)

# Генерация слоя в пользовательской системе координат
layer = generator.generate_layer(x_num=11, y_num=11, layer_cs=custom_cs)
```

### Пользовательские Условия Соединения

```python
def custom_connect_condition(point):
    # Пользовательская логика для фильтрации точек пересечения
    return np.linalg.norm(point) < 10.0

faces = generator.compute_faces(connect_condition=custom_connect_condition)
```

## Справочник API

### GeometryConfig

Класс конфигурации для геометрических параметров.

```python
config = GeometryConfig(
    radius: float,           # Базовый радиус шестигранной призмы
    height: float,           # Высота призмы
    bev_angle: float,        # Угол скоса в радианах
    size_trick: float = 1.1  # Масштабирующий коэффициент для генерации геометрии
)
```

### LayerGenerator

Главный фасадный класс для генерации STL слоев.

```python
generator = LayerGenerator(
    config: GeometryConfig,
    bend_radius: Optional[float] = None
)

# Методы
layer = generator.generate_layer(x_num: int, y_num: int)
bent_layer = generator.apply_bending(bend_radius: float)
faces = generator.compute_faces(connect_condition: Optional[Callable] = None)
generator.export_stl(filename: str, solid_name: str = "bf8_sls", format: str = "ascii")
```

## Тестирование

Запуск набора тестов:

```bash
cd models/stlgen
python -m pytest tests/
```

Или запуск отдельных файлов тестов:

```bash
python tests/test_integration.py
```

## Миграция с Устаревшего Кода

Новый ООП интерфейс заменяет устаревший процедурный код:

### Устаревший Код
```python
# Старый процедурный подход
import generate_unit_block_FOR_TESTS as gub_TESTS
import Layer_configuraton as l_build
import bev_transform as bev_tr
import STL_Generator as stlG

# Сложный процедурный пайплайн...
```

### Новый ООП Код
```python
# Новый объектно-ориентированный подход
from models.stlgen import GeometryConfig, LayerGenerator

config = GeometryConfig(radius=4.0, height=1.0, bev_angle=0.5)
generator = LayerGenerator(config)
layer = generator.generate_complete_layer(x_num=11, y_num=11)
```

## Структура Файлов

```
models/stlgen/
├── __init__.py                 # Инициализация пакета
├── main.py                     # Главная точка входа
├── layer_generator.py          # Главный фасадный класс
├── geometry/                   # Классы геометрии
│   ├── __init__.py
│   ├── config.py              # GeometryConfig
│   ├── coordinate_system.py   # CoordinateSystem
│   ├── beveled_prism.py       # BeveledPrism
│   ├── prism_layer.py         # PrismLayer
│   └── convex_solver.py       # ConvexHullSolver
├── transforms/                 # Классы трансформаций
│   ├── __init__.py
│   └── cylindrical.py         # CylindricalTransform
├── export/                     # Классы экспорта
│   ├── __init__.py
│   └── stl_exporter.py        # STLExporter
└── tests/                      # Набор тестов
    ├── __init__.py
    └── test_integration.py     # Интеграционные тесты
```

## Вклад в Проект

1. Следуйте существующему стилю кода и паттернам
2. Добавляйте аннотации типов ко всем новым функциям и методам
3. Пишите тесты для новой функциональности
4. Обновляйте документацию по мере необходимости

## Лицензия

Этот проект является частью системы разработки SLS.
