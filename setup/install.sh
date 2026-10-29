#!/usr/bin/env bash
# Installs Freeroom: Python env (uv) with the right PyTorch build, the models
# and a desktop launcher. Safe to re-run.
#
#   setup/install.sh            auto: distrobox if installed, else native
#   setup/install.sh --box      in the "freeroom" distrobox (Fedora; Bazzite etc.)
#   setup/install.sh --native   directly on this system
#   setup/install.sh --cpu      force the CPU PyTorch build (also: --cu128)
#
# PyTorch: CUDA 12.8 build when an NVIDIA GPU is found (nvidia-smi), otherwise
# the much smaller CPU build. The choice is saved in .install for freeroom.sh.
#
# macOS (Apple Silicon only): always native; the "cpu" build there is the
# standard macOS PyTorch, which uses the Apple GPU through Metal.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
BOX="${FREEROOM_BOX:-freeroom}"
MODE=auto
EXTRA=""
for arg in "$@"; do
  case "$arg" in
    --box) MODE=box ;;
    --native) MODE=native ;;
    --cpu) EXTRA=cpu ;;
    --cu128) EXTRA=cu128 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$(uname -s)" = Darwin ]; then
  if [ "$(uname -m)" != arm64 ]; then
    echo "Intel Macs aren't supported: PyTorch 2.8 is only built for Apple Silicon (M1 or newer)." >&2
    exit 1
  fi
  if [ "$MODE" = box ] || [ "$EXTRA" = cu128 ]; then
    echo "On macOS only a native install with the Metal (cpu) build is possible." >&2
    exit 2
  fi
  MODE=native
  EXTRA=cpu
  echo "Apple Silicon Mac: PyTorch with Metal GPU support."
fi

if [ -z "$EXTRA" ]; then
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    EXTRA=cu128
    echo "NVIDIA GPU found: $(nvidia-smi -L | head -1)"
  else
    EXTRA=cpu
    echo "No NVIDIA GPU found: using the CPU build (AI steps will be slow)."
  fi
fi
if [ "$MODE" = auto ]; then
  if command -v distrobox >/dev/null; then MODE=box; else MODE=native; fi
fi
echo "Installing: $MODE, PyTorch $EXTRA"

# uv sync with the chosen build; the extra decides which PyTorch index is used.
SYNC="command -v uv >/dev/null || [ -x \$HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=\$HOME/.local/bin:\$PATH
  cd '$HERE' && uv sync --extra $EXTRA"

if [ "$MODE" = box ]; then
  if ! distrobox list | awk -F'|' '{gsub(/ /,"",$2); print $2}' | grep -qx "$BOX"; then
    echo "Creating distrobox '$BOX'…"
    NV=""
    [ "$EXTRA" = cu128 ] && NV=--nvidia
    # shellcheck disable=SC2086  # NV is empty or one flag
    distrobox create --name "$BOX" --image registry.fedoraproject.org/fedora-toolbox:latest $NV --yes
  fi
  echo "Installing Qt runtime libs inside the box…"
  # The pip PySide6 wheel brings Qt itself; these pull in the X11/Wayland/GL
  # system libraries its platform plugins need.
  distrobox enter "$BOX" -- sudo dnf install -y --setopt=install_weak_deps=False \
    qt6-qtbase-gui qt6-qtwayland xcb-util-cursor mesa-libGL mesa-libEGL curl
  echo "Installing uv + Python deps…"
  distrobox enter "$BOX" -- bash -lc "$SYNC"
  UV='~/.local/bin/uv'  # inside the box
else
  echo "Installing uv + Python deps…"
  bash -lc "$SYNC"
  # wherever uv is (~/.local/bin, Homebrew…): app launchers get a minimal PATH
  UV="$(bash -lc 'export PATH=$HOME/.local/bin:$PATH; command -v uv')"
fi

printf 'mode=%s\nbox=%s\nextra=%s\nuv=%s\n' "$MODE" "$BOX" "$EXTRA" "$UV" > "$HERE/.install"

"$HERE/setup/download-models.sh"
"$HERE/setup/install-launcher.sh"

echo
if [ "$(uname -s)" = Darwin ]; then WHERE="~/Applications (Freeroom)"; else WHERE="the app menu (Freeroom)"; fi
echo "Done. Start it from $WHERE or: $HERE/freeroom.sh [photo]"
