#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT="${APP_ROOT:-/srv/vocaptest}"
APP_DIR="${APP_DIR:-$APP_ROOT/app}"
VENV_DIR="${VENV_DIR:-$APP_ROOT/venv}"
REPO_URL="${REPO_URL:-https://github.com/Loong-C/VocaPTest.git}"
BRANCH="${BRANCH:-master}"
SERVICE_NAME="${SERVICE_NAME:-vocaptest}"
SERVICE_USER="${SERVICE_USER:-vocaptest}"
VITE_BASE_PATH="${VITE_BASE_PATH:-/VocaPTest/}"
NGINX_SITE="${NGINX_SITE:-/etc/nginx/sites-available/bookstore}"
NGINX_SNIPPET="${NGINX_SNIPPET:-/etc/nginx/snippets/vocaptest-locations.conf}"
NGINX_SECURITY_SNIPPET="${NGINX_SECURITY_SNIPPET:-/etc/nginx/snippets/vocaptest-security-headers.conf}"
NGINX_RATE_LIMIT_CONF="${NGINX_RATE_LIMIT_CONF:-/etc/nginx/conf.d/vocaptest-rate-limit.conf}"
SKIP_SYSTEM_PACKAGES="${SKIP_SYSTEM_PACKAGES:-0}"
SKIP_PYTHON_DEPS="${SKIP_PYTHON_DEPS:-0}"
SKIP_SERVICE_INSTALL="${SKIP_SERVICE_INSTALL:-0}"
SKIP_NGINX_INSTALL="${SKIP_NGINX_INSTALL:-0}"

export DEBIAN_FRONTEND=noninteractive

log() {
  printf '[vocaptest-deploy] %s\n' "$*"
}

require_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "Run this script as root." >&2
    exit 1
  fi
}

install_system_packages() {
  log "Installing system packages"
  apt-get update
  apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libsndfile1 \
    nginx \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv
}

sync_repo() {
  log "Syncing $REPO_URL ($BRANCH) into $APP_DIR"
  mkdir -p "$APP_ROOT"
  if ! git config --global --get-all safe.directory | grep -Fxq "$APP_DIR"; then
    git config --global --add safe.directory "$APP_DIR"
  fi

  if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" fetch origin "$BRANCH"
    git -C "$APP_DIR" checkout "$BRANCH"
    git -C "$APP_DIR" reset --hard "origin/$BRANCH"
  else
    rm -rf "$APP_DIR"
    git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
  fi
}

install_python_deps() {
  log "Installing Python dependencies"
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

  local filtered_requirements
  filtered_requirements="$(mktemp)"
  grep -Ev '^(torch|torchaudio)([<>=~! ].*)?$' "$APP_DIR/requirements.txt" > "$filtered_requirements"

  "$VENV_DIR/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio
  "$VENV_DIR/bin/python" -m pip install -r "$filtered_requirements"
  "$VENV_DIR/bin/python" -m pip install -e "$APP_DIR"
  rm -f "$filtered_requirements"
}

build_frontend() {
  log "Building frontend with VITE_BASE_PATH=$VITE_BASE_PATH"
  cd "$APP_DIR/web"
  npm ci
  VITE_BASE_PATH="$VITE_BASE_PATH" npm run build
}

install_service() {
  log "Installing systemd service"
  if ! getent passwd "$SERVICE_USER" >/dev/null; then
    useradd --system --home "$APP_ROOT" --shell /usr/sbin/nologin "$SERVICE_USER"
  fi

  install -m 0644 "$APP_DIR/deploy/vocaptest.service" "/etc/systemd/system/$SERVICE_NAME.service"
  mkdir -p "$APP_ROOT/shared/huggingface"
  chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_ROOT"
  systemctl daemon-reload
  systemctl enable "$SERVICE_NAME.service"
  systemctl restart "$SERVICE_NAME.service"
}

install_nginx() {
  log "Installing Nginx snippets"
  install -m 0644 "$APP_DIR/deploy/nginx-vocaptest-security-headers.conf" "$NGINX_SECURITY_SNIPPET"
  install -m 0644 "$APP_DIR/deploy/nginx-vocaptest-rate-limit.conf" "$NGINX_RATE_LIMIT_CONF"
  install -m 0644 "$APP_DIR/deploy/nginx-vocaptest-locations.conf" "$NGINX_SNIPPET"

  if [ ! -f "$NGINX_SITE" ]; then
    echo "Expected Nginx site file not found: $NGINX_SITE" >&2
    exit 1
  fi

  if ! grep -Fq "include $NGINX_SNIPPET;" "$NGINX_SITE"; then
    log "Adding snippet include to $NGINX_SITE"
    SITE_PATH="$NGINX_SITE" SNIPPET_PATH="$NGINX_SNIPPET" python3 - <<'PY'
from pathlib import Path
import os

site = Path(os.environ["SITE_PATH"])
snippet = os.environ["SNIPPET_PATH"]
text = site.read_text()
include_line = f"    include {snippet};\n"
if include_line.strip() not in text:
    marker = "    client_max_body_size 20m;\n"
    if marker in text:
        text = text.replace(marker, marker + "\n" + include_line, 1)
    else:
        server_marker = "    server_name linkukai.com www.linkukai.com 187.77.136.20;\n"
        if server_marker not in text:
            raise SystemExit("Could not find an insertion point in the Nginx site file.")
        text = text.replace(server_marker, server_marker + "\n" + include_line, 1)
    backup = site.with_suffix(site.suffix + ".bak")
    backup.write_text(site.read_text())
    site.write_text(text)
PY
  fi

  nginx -t
  systemctl reload nginx
}

main() {
  require_root
  if [ "$SKIP_SYSTEM_PACKAGES" = "1" ]; then
    log "Skipping system package install"
  else
    install_system_packages
  fi

  sync_repo

  if [ "$SKIP_PYTHON_DEPS" = "1" ]; then
    log "Skipping Python dependency install"
  else
    install_python_deps
  fi

  build_frontend

  if [ "$SKIP_SERVICE_INSTALL" = "1" ]; then
    log "Skipping systemd service install/restart"
  else
    install_service
  fi

  if [ "$SKIP_NGINX_INSTALL" = "1" ]; then
    log "Skipping Nginx snippet install/reload"
  else
    install_nginx
  fi

  log "Done"
}

main "$@"
