#!/usr/bin/env bash
set -euo pipefail

# vllmtop uninstaller — removes binary from known locations
# Usage: ./uninstall.sh [--prefix DIR]

PREFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --user) PREFIX="$HOME/.local/bin"; shift ;;
    --system) PREFIX="/usr/local/bin"; shift ;;
    -h|--help)
      echo "Usage: $0 [--user|--system] [--prefix DIR]"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

targets=()
if [[ -n "$PREFIX" ]]; then
  targets=("$PREFIX/vllmtop")
else
  targets=("/usr/local/bin/vllmtop" "$HOME/.local/bin/vllmtop")
fi

removed=0
for t in "${targets[@]}"; do
  if [[ -f "$t" ]]; then
    rm -f "$t"
    echo "removed $t"
    removed=1
  fi
done

if [[ $removed -eq 0 ]]; then
  echo "vllmtop not found in ${targets[*]}"
fi
