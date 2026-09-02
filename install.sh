#!/usr/bin/env bash
set -euo pipefail

# vllmtop installer — copies single binary to /usr/local/bin or ~/.local/bin
# Also supports one-line remote install: curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | bash
# Usage:
#   ./install.sh              # auto: /usr/local/bin if writable else ~/.local/bin
#   ./install.sh --user       # ~/.local/bin
#   ./install.sh --system     # /usr/local/bin (requires sudo if not root)
#   ./install.sh --prefix /opt/bin
#   VLLMTOP_VERSION=v0.1.0 ./install.sh   # specific version

PREFIX=""
MODE="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) MODE="user"; shift ;;
    --system) MODE="system"; shift ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --socket) shift 2 ;; # ignored
    -h|--help)
      echo "Usage: $0 [--user|--system] [--prefix DIR]"
      echo "  --user    install to ~/.local/bin"
      echo "  --system  install to /usr/local/bin"
      echo "  --prefix  custom directory"
      echo "Env: VLLMTOP_VERSION=v0.1.0 to pin version (default: latest)"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Locate binary: prefer ./vllmtop, fallback to target/release/vllmtop, else download
SRC=""
TMP_DOWNLOAD=""
cleanup() { [[ -n "$TMP_DOWNLOAD" && -f "$TMP_DOWNLOAD" ]] && rm -f "$TMP_DOWNLOAD"; }
trap cleanup EXIT

if [[ -f "./vllmtop" ]]; then
  SRC="./vllmtop"
elif [[ -f "./target/release/vllmtop" ]]; then
  SRC="./target/release/vllmtop"
else
  # No local binary — download from GitHub Releases (for one-line install)
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64|amd64) ASSET="vllmtop-linux-x86_64" ;;
    aarch64|arm64) ASSET="vllmtop-linux-aarch64" ;;
    *) echo "error: unsupported arch $ARCH (supported: x86_64, aarch64)" >&2; exit 1 ;;
  esac
  if [[ -n "${VLLMTOP_VERSION:-}" ]]; then
    URL="https://github.com/hanana-kick/vllmtop/releases/download/${VLLMTOP_VERSION}/${ASSET}"
  else
    URL="https://github.com/hanana-kick/vllmtop/releases/latest/download/${ASSET}"
  fi
  TMP_DOWNLOAD="$(mktemp /tmp/vllmtop.XXXXXX)"
  echo "downloading $URL ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$TMP_DOWNLOAD" "$URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$TMP_DOWNLOAD" "$URL"
  else
    echo "error: need curl or wget to download binary" >&2
    echo "Alternatively, download manually from https://github.com/hanana-kick/vllmtop/releases" >&2
    exit 1
  fi
  chmod +x "$TMP_DOWNLOAD"
  SRC="$TMP_DOWNLOAD"
fi

# Determine destination
DEST_DIR=""
if [[ -n "$PREFIX" ]]; then
  DEST_DIR="$PREFIX"
elif [[ "$MODE" == "user" ]]; then
  DEST_DIR="$HOME/.local/bin"
elif [[ "$MODE" == "system" ]]; then
  DEST_DIR="/usr/local/bin"
else
  # auto
  if [[ -w "/usr/local/bin" ]] || [[ $EUID -eq 0 ]]; then
    DEST_DIR="/usr/local/bin"
  else
    DEST_DIR="$HOME/.local/bin"
  fi
fi

mkdir -p "$DEST_DIR"
install -m 755 "$SRC" "$DEST_DIR/vllmtop"
echo "installed $SRC -> $DEST_DIR/vllmtop"

# Ensure ~/.local/bin in PATH hint
if [[ "$DEST_DIR" == "$HOME/.local/bin" ]]; then
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) echo "hint: add to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
  esac
fi

# Docker socket permission hint
if [[ -S "/var/run/docker.sock" ]]; then
  if ! docker ps >/dev/null 2>&1; then
    echo "note: 'docker ps' failed — you may need to add yourself to the docker group:"
    echo "  sudo usermod -aG docker \$USER && newgrp docker"
    echo "or run with sudo: sudo vllmtop"
  fi
else
  echo "note: /var/run/docker.sock not found — set --socket if Docker uses a different path"
fi

echo "run: vllmtop            # interactive selector"
echo "     vllmtop <name>     # attach directly"
