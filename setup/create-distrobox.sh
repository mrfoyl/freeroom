#!/usr/bin/env bash
# Creates the "freeroom" distrobox (Fedora + NVIDIA driver passthrough),
# installs the Python env with CUDA PyTorch, downloads the models and adds a
# desktop launcher. Run on the Bazzite host. Safe to re-run.
set -euo pipefail

BOX="${1:-freeroom}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if ! distrobox list | awk -F'|' '{gsub(/ /,"",$2); print $2}' | grep -qx "$BOX"; then
  echo "Creating distrobox '$BOX'…"
  distrobox create --name "$BOX" --image registry.fedoraproject.org/fedora-toolbox:latest --nvidia --yes
fi

echo "Installing Qt runtime libs inside the box…"
# The pip PySide6 wheel brings Qt itself; these pull in the X11/Wayland/GL
# system libraries its platform plugins need.
distrobox enter "$BOX" -- sudo dnf install -y --setopt=install_weak_deps=False \
  qt6-qtbase-gui qt6-qtwayland xcb-util-cursor mesa-libGL mesa-libEGL curl

echo "Installing uv + Python deps (CUDA PyTorch, ~3 GB first time)…"
distrobox enter "$BOX" -- bash -lc "
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=\$HOME/.local/bin:\$PATH
  cd '$HERE' && uv sync
"

"$HERE/setup/download-models.sh"
"$HERE/setup/install-launcher.sh" "$BOX"

echo
echo "Done. Start it from the app menu (Freeroom) or: $HERE/freeroom.sh [photo]"
