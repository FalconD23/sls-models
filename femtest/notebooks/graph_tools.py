import numpy as np
import matplotlib.pyplot as plt


def plot_multi_curves(
    x_datasets,
    y_datasets,
    curve_labels=None,        # подписи отдельных кривых (легенда)
    colors=None,              # список цветов или None
    markers=None,             # список маркеров или None
    linewidth=2.0,
    markersize=6,
    xlabel="Ось X",
    ylabel="Ось Y",
    plot_title="График",
    legend_title=None,        # заголовок легенды (подпись графика)
    alpha=0.8,
    show_grid=True,
    show_stats=False,
    x_tick_rotation=45,
    figsize=(12, 7),
):
    """
    Универсальная отрисовка нескольких кривых на одном графике.

    Параметры:
    - x_datasets: list[array-like]   -> наборы X для каждой кривой
    - y_datasets: list[array-like]   -> наборы Y для каждой кривой
    - curve_labels: list[str]        -> подписи кривых в легенде
    - colors: list                   -> цвета кривых
    - markers: list[str]             -> маркеры кривых
    - linewidth: float               -> толщина линий
    - xlabel, ylabel, plot_title     -> подписи и заголовок
    - legend_title: str              -> заголовок блока легенды
    - show_stats: bool               -> печатать min/max/mean/std по каждой кривой
    """
    if len(x_datasets) != len(y_datasets):
        raise ValueError("Количество наборов x_datasets и y_datasets должно совпадать.")

    n = len(x_datasets)
    if n == 0:
        raise ValueError("Передайте хотя бы один набор данных.")

    # Значения по умолчанию
    if curve_labels is None:
        curve_labels = [f"Кривая {i+1}" for i in range(n)]
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n))
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    if len(curve_labels) != n:
        raise ValueError("Длина curve_labels должна совпадать с числом кривых.")

    plt.figure(figsize=figsize)

    all_x = set()
    for i, (x, y) in enumerate(zip(x_datasets, y_datasets)):
        x = np.asarray(x)
        y = np.asarray(y)

        if len(x) != len(y):
            raise ValueError(f"Длины x и y не совпадают для кривой #{i+1}.")

        all_x.update(x.tolist())

        plt.plot(
            x,
            y,
            marker=markers[i % len(markers)],
            linewidth=linewidth,
            markersize=markersize,
            color=colors[i % len(colors)],
            label=curve_labels[i],
            alpha=alpha,
        )

    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(plot_title, fontsize=14, fontweight='bold')

    if show_grid:
        plt.grid(True, alpha=0.3, linestyle='--')

    plt.legend(loc='best', fontsize=10, framealpha=0.9, title=legend_title)

    # Показываем все уникальные X-тиковые значения (если числовые)
    try:
        all_x_sorted = sorted(all_x)
        plt.xticks(all_x_sorted, rotation=x_tick_rotation)
    except TypeError:
        pass  # если X не сортируются (например, смешанные типы)

    plt.tight_layout()
    plt.show()

    # Опциональная статистика
    if show_stats:
        print("\n" + "=" * 60)
        print("СТАТИСТИКА ПО КРИВЫМ:")
        print("=" * 60)
        for i, y in enumerate(y_datasets):
            y = np.asarray(y)
            x = np.asarray(x_datasets[i])

            min_idx = np.argmin(y)
            max_idx = np.argmax(y)

            print(f"\n{curve_labels[i]}")
            print(f"  Минимум: {y[min_idx]:.4f} при X={x[min_idx]}")
            print(f"  Максимум: {y[max_idx]:.4f} при X={x[max_idx]}")
            print(f"  Среднее: {np.mean(y):.4f}")
            print(f"  Стандартное отклонение: {np.std(y):.4f}")