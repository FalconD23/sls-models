# Docker Guide — SLS-Models FreeCAD

Контейнер упаковывает FreeCAD + CalculiX + Python окружение.
GUI доступен в браузере через **noVNC** — работает локально и в облаке без установки VNC-клиента.

## Структура файлов

```
sls-models/
├── Dockerfile            ← образ Ubuntu 22.04 + FreeCAD + CalculiX + XFCE + noVNC
├── docker-compose.yml    ← оркестрация: порты, volumes, env
├── docker/
│   └── entrypoint.sh     ← запуск Xvfb → XFCE → VNC → noVNC
├── .dockerignore         ← что не копировать в образ
└── DOCKER_GUIDE.md       ← этот файл
```

---

## Требования

- **Docker** >= 24.x
- **Docker Compose** >= 2.x
- Порты **5900** и **6080** должны быть свободны (или измени в `docker-compose.yml`)

---

## Инструкция запуска

### Шаг 1 — Сборка образа

Выполни один раз (или после изменения `requirements.txt`/`Dockerfile`):

```bash
cd /home/ubnps23/tecHub/SLS_dev/sls-models
docker compose build
```

> Сборка занимает ~10–20 минут (скачивает FreeCAD, CalculiX, Python-пакеты).

---

### Шаг 2а — Запуск с GUI (интерактивная работа)

```bash
docker compose up -d
```

Затем открой в браузере:
```
http://localhost:6080/vnc.html
```

**Пароль VNC:** `freecad1234`

Откроется рабочий стол XFCE. Запусти FreeCAD из терминала внутри:
```bash
freecad /workspace/femtest/some_model.FCStd
```

---

### Шаг 2б — Batch-расчёт без GUI (быстро, для облака)

```bash
docker compose run --rm freecad-sls \
    freecad -c /workspace/femtest/optimize_angle.py
```

Или несколько параллельных расчётов:

```bash
# Терминал 1
docker compose run --rm freecad-sls \
    freecad -c /workspace/femtest/optimize_angle.py

# Терминал 2 (другой набор параметров)
docker compose run --rm freecad-sls \
    freecad -c /workspace/femtest/main_file_list.py
```

---

### Шаг 3 — Остановка контейнера

```bash
docker compose down
```

---

## Облачный запуск (AWS / GCP / VPS)

1. Клонируй репозиторий на сервер:
   ```bash
   git clone <repo_url> sls-models
   cd sls-models
   ```

2. Открой порт `6080` в Security Group / Firewall сервера.

3. Собери и запусти:
   ```bash
   docker compose build
   docker compose up -d
   ```

4. Открой в браузере:
   ```
   http://<SERVER_IP>:6080/vnc.html
   ```

> Для безопасности в production смени пароль VNC: отредактируй `Dockerfile`,
> строку `x11vnc -storepasswd freecad1234 ...` → замени `freecad1234` на свой пароль.

---

## Проверка работоспособности

После запуска контейнера (`docker compose up -d`) выполни в новом терминале:

```bash
# 1. Проверить что контейнер запущен
docker ps | grep sls_freecad_gui

# 2. Проверить FreeCAD
docker exec -it sls_freecad_gui \
    freecad --version

# 3. Проверить CalculiX
docker exec -it sls_freecad_gui \
    ccx --help

# 4. Проверить Python-окружение и ключевые пакеты
docker exec -it sls_freecad_gui \
    /home/sls_user/my_env_freecad/bin/python3 -c \
    "import numpy, scipy, yaml, femtools; print('OK — numpy scipy yaml femtools все импортированы')"

# 5. Проверить пути (должен распечатать config_path без ошибок)
docker exec -it sls_freecad_gui bash -c \
    "cd /workspace && freecad -c femtest/main.py --check-paths 2>&1 | head -20"

# 6. Посмотреть логи VNC/noVNC
docker exec -it sls_freecad_gui cat /tmp/x11vnc.log
docker exec -it sls_freecad_gui cat /tmp/novnc.log
```

---

## Часто используемые команды

| Действие | Команда |
|---|---|
| Войти в bash контейнера | `docker exec -it sls_freecad_gui bash` |
| Посмотреть логи запуска | `docker compose logs -f` |
| Перезапустить контейнер | `docker compose restart` |
| Остановить и удалить | `docker compose down` |
| Пересобрать образ | `docker compose build --no-cache` |
| Запустить скрипт в batch | `docker compose run --rm freecad-sls freecad -c /workspace/femtest/optimize_angle.py` |

---

## Результаты расчётов

FEM-результаты FreeCAD/CalculiX сохраняются в `/tmp/fcfem_*` внутри контейнера.
Чтобы сохранять их на хосте, добавь volume-маунт в `docker-compose.yml`:

```yaml
volumes:
  - ./fem_results:/tmp:rw
```

---

## Изменение параметров (пути, структуры)

Все пути в Python-скриптах теперь определяются автоматически относительно расположения скрипта (`Path(__file__).resolve().parent`). Файл `main_config.yaml` содержит `root_dir: "__AUTO__"` — это означает, что путь вычисляется автоматически и работает как локально, так и в Docker.

Если нужно переключиться на другую структуру (например с `bev_trunc_octahedrons` на `bev_hex_prisms_plate`), измени строку `config_path` в нужном Python-скрипте.







