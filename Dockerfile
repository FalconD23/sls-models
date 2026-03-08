FROM ubuntu:22.04

# ──────────────────────────────────────────────
# 1. Системные зависимости
# ──────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    DISPLAY=:1 \
    VNC_PORT=5900 \
    NOVNC_PORT=6080 \
    VNC_RESOLUTION=1920x1080 \
    VNC_COL_DEPTH=24

RUN apt-get update && apt-get install -y \
    # Locale
    locales \
    && locale-gen en_US.UTF-8 \
    && apt-get install -y \
    # Core tools
    software-properties-common \
    wget curl git nano ca-certificates \
    # Python
    python3 python3-pip python3-venv \
    # GUI / Desktop
    xfce4 xfce4-terminal xfce4-taskmanager \
    # Virtual framebuffer + VNC
    xvfb x11vnc \
    # noVNC (web-based VNC client)
    novnc websockify \
    # Fonts
    fonts-liberation fonts-dejavu \
    # CalculiX FEM solver
    calculix-ccx \
    && add-apt-repository ppa:freecad-maintainers/freecad-stable \
    && apt-get update \
    && apt-get install -y freecad \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────────
# 2. Пользователь (не root)
# ──────────────────────────────────────────────
RUN useradd -ms /bin/bash sls_user
USER sls_user
WORKDIR /home/sls_user

# ──────────────────────────────────────────────
# 3. Python виртуальное окружение
# ──────────────────────────────────────────────
RUN python3 -m venv /home/sls_user/my_env_freecad
ENV PATH="/home/sls_user/my_env_freecad/bin:$PATH"

# ──────────────────────────────────────────────
# 4. Python зависимости (кэш отдельным слоем)
# ──────────────────────────────────────────────
COPY --chown=sls_user:sls_user femtest/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# ──────────────────────────────────────────────
# 5. Проект
# ──────────────────────────────────────────────
COPY --chown=sls_user:sls_user . /workspace/
WORKDIR /workspace

# ──────────────────────────────────────────────
# 6. VNC пароль
# ──────────────────────────────────────────────
RUN mkdir -p /home/sls_user/.vnc && \
    x11vnc -storepasswd freecad1234 /home/sls_user/.vnc/passwd

# ──────────────────────────────────────────────
# 7. Entrypoint
# ──────────────────────────────────────────────
COPY --chown=sls_user:sls_user docker/entrypoint.sh /home/sls_user/entrypoint.sh
RUN chmod +x /home/sls_user/entrypoint.sh

# ──────────────────────────────────────────────
# 8. Порты
# ──────────────────────────────────────────────
# 5900 — VNC (прямое подключение через VNC-клиент)
# 6080 — noVNC (браузер)
EXPOSE 5900 6080

ENTRYPOINT ["/home/sls_user/entrypoint.sh"]







