#!/usr/bin/env bash
set -euo pipefail

# vllmtop installer — copies single binary to /usr/local/bin or ~/.local/bin
# Usage:
#   ./install.sh              # auto: /usr/local/bin if writable else ~/.local/bin
#   ./install.sh --user       # ~/.local/bin
#   ./install.sh --system     # /usr/local/bin (requires sudo if not root)
#   ./install.sh --prefix /opt/bin
#   ./install.sh --socket /var/run/docker.sock  # no effect, just for docs

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
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Locate binary: prefer ./vllmtop, fallback to target/release/vllmtop
SRC=""
if [[ -f "./vllmtop" ]]; then
  SRC="./vllmtop"
elif [[ -f "./target/release/vllmtop" ]]; then
  SRC="./target/release/vllmtop"
else
  echo "error: binary not found. Build first:" >&2
  echo "  cargo build --release" >&2
  echo "  cp target/release/vllmtop ./vllmtop" >&2
  exit 1
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
