#!/bin/bash
# =============================================================================
# Entrypoint for sls-models FreeCAD Docker container
# Starts: Xvfb → XFCE4 desktop → x11vnc → noVNC (web)
# =============================================================================
set -e

export DISPLAY=:1
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
VNC_RESOLUTION="${VNC_RESOLUTION:-1920x1080}"
VNC_COL_DEPTH="${VNC_COL_DEPTH:-24}"
VNC_PASSWD_FILE="/home/sls_user/.vnc/passwd"

echo "========================================================"
echo "  SLS-Models FreeCAD Container"
echo "========================================================"

# ── 1. Виртуальный дисплей ────────────────────────────────
echo "[1/4] Starting Xvfb virtual display (${VNC_RESOLUTION}x${VNC_COL_DEPTH})..."
Xvfb :1 -screen 0 "${VNC_RESOLUTION}x${VNC_COL_DEPTH}" -ac +extension GLX +render -noreset &
XVFB_PID=$!
sleep 2

# ── 2. Рабочий стол XFCE ─────────────────────────────────
echo "[2/4] Starting XFCE4 desktop..."
startxfce4 &
sleep 3

# Disable lock screen for unattended noVNC sessions in local/cloud runs.
xfconf-query -c xfce4-screensaver -p /lock/enabled -n -t bool -s false || true
xfconf-query -c xfce4-screensaver -p /saver/enabled -n -t bool -s false || true
pkill -f xfce4-screensaver || true

# ── 3. VNC сервер ─────────────────────────────────────────
echo "[3/4] Starting x11vnc on port ${VNC_PORT}..."
x11vnc \
    -display :1 \
    -rfbport "${VNC_PORT}" \
    -rfbauth "${VNC_PASSWD_FILE}" \
    -forever \
    -noxdamage \
    -repeat \
    -shared \
    -logfile /tmp/x11vnc.log &
sleep 1

# ── 4. noVNC (доступ через браузер) ───────────────────────
echo "[4/4] Starting noVNC on port ${NOVNC_PORT}..."
websockify \
    --web=/usr/share/novnc/ \
    --log-file=/tmp/novnc.log \
    "${NOVNC_PORT}" \
    "localhost:${VNC_PORT}" &
NOVNC_PID=$!
sleep 1

# ── Готово ─────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  GUI READY"
echo "  Browser (noVNC): http://CONTAINER_IP:${NOVNC_PORT}/vnc.html"
echo "  VNC client:      CONTAINER_IP:${VNC_PORT}"
echo "  VNC password:    freecad1234"
echo ""
echo "  Workspace:       /workspace"
echo "  Python env:      /home/sls_user/my_env_freecad"
echo ""
echo "  Batch FEM run (no GUI):"
echo "    freecad -c /workspace/femtest/optimize_angle.py"
echo "========================================================"
echo ""

# ── Запускаем команду (если передана) или держим контейнер ─
if [ "$#" -gt 0 ]; then
    echo "Running command: $*"
    exec "$@"
else
    # Держим контейнер живым, выводим логи VNC
    tail -f /tmp/x11vnc.log /tmp/novnc.log 2>/dev/null || tail -f /dev/null
fi







