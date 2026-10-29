#!/usr/bin/env bash
# Installs Freeroom: Python env (uv) with the right PyTorch build, the models
# and a desktop launcher. Safe to re-run.
#
#   setup/install.sh            auto: distrobox if installed, else native
#   setup/install.sh --box      in the "freeroom" distrobox (Fedora; Bazzite etc.)
#   setup/install.sh --native   directly on this system
#   setup/install.sh --cpu      force the CPU PyTorch build (also: --cu128, --rocm)
#
# PyTorch: CUDA 12.8 build when an NVIDIA GPU is found (nvidia-smi), ROCm 6.4
# when an AMD GPU that build supports is found (from sysfs, no ROCm tools
# needed), otherwise the much smaller CPU build. The choice is saved in
# .install for freeroom.sh.
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
    --rocm) EXTRA=rocm ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$(uname -s)" = Darwin ]; then
  if [ "$(uname -m)" != arm64 ]; then
    echo "Intel Macs aren't supported: PyTorch 2.8 is only built for Apple Silicon (M1 or newer)." >&2
    exit 1
  fi
  if [ "$MODE" = box ] || [ "$EXTRA" = cu128 ] || [ "$EXTRA" = rocm ]; then
    echo "On macOS only a native install with the Metal (cpu) build is possible." >&2
    exit 2
  fi
  MODE=native
  EXTRA=cpu
  echo "Apple Silicon Mac: PyTorch with Metal GPU support."
fi

# AMD GPUs (gfx names) the ROCm 6.4 PyTorch build really runs on: its own
# kernels and rocBLAS's both (gfx900/906 have PyTorch kernels but no rocBLAS)
ROCM_TARGETS="gfx908 gfx90a gfx942 gfx1030 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201"
amd_gpus() {  # gfx name of each AMD GPU the kernel driver exposes for compute
  local f v
  for f in "${FREEROOM_KFD_NODES:-/sys/class/kfd/kfd/topology/nodes}"/*/properties; do
    [ -f "$f" ] || continue
    v="$(sed -n 's/^gfx_target_version //p' "$f")"
    # e.g. 110000 -> gfx1100, 90012 -> gfx90c (major, then minor and stepping in hex)
    [ -n "$v" ] && [ "$v" != 0 ] && printf 'gfx%d%x%x\n' $((v / 10000)) $((v / 100 % 100)) $((v % 100))
  done
  return 0
}

if [ -z "$EXTRA" ]; then
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    EXTRA=cu128
    echo "NVIDIA GPU found: $(nvidia-smi -L | head -1)"
  else
    EXTRA=cpu
    AMD=""
    for g in $(amd_gpus); do
      AMD="${AMD:-$g}"
      case " $ROCM_TARGETS " in *" $g "*) EXTRA=rocm; AMD=$g; break ;; esac
    done
    if [ "$EXTRA" = rocm ]; then
      echo "AMD GPU found ($AMD): using the ROCm build."
    elif [ -n "$AMD" ]; then
      echo "AMD GPU found ($AMD), but the ROCm PyTorch build doesn't support it: using the CPU build."
      echo "  (To try anyway: --rocm, with HSA_OVERRIDE_GFX_VERSION set to a close supported target.)"
    else
      echo "No NVIDIA or supported AMD GPU found: using the CPU build (AI steps will be slow)."
    fi
  fi
fi
if [ "$EXTRA" = rocm ] && [ -e /dev/kfd ] && ! { [ -r /dev/kfd ] && [ -w /dev/kfd ]; }; then
  echo "Note: you can't use /dev/kfd yet; ROCm needs it (the 'render' group):" >&2
  echo "  sudo usermod -aG render,video \"\$USER\"   (then log out and back in)" >&2
fi
if [ "$MODE" = auto ]; then
  if command -v distrobox >/dev/null; then MODE=box; else MODE=native; fi
fi
echo "Installing: $MODE, PyTorch $EXTRA"

# System packages a native Linux install needs: curl for the downloads, and
# the libraries Qt's display plugins load (PySide6 brings Qt itself). Only the
# missing ones are installed, with sudo. "a|b" = either name (differs by release).
APT_PKGS="curl ca-certificates libglib2.0-0t64|libglib2.0-0 libdbus-1-3 libfontconfig1
  libfreetype6 libegl1 libgl1 libdrm2 libx11-6 libx11-xcb1 libxkbcommon0 libxkbcommon-x11-0
  libxcb1 libxcb-cursor0 libxcb-glx0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0
  libxcb-render0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-util1
  libxcb-xfixes0 libxcb-xkb1 libwayland-client0 libwayland-cursor0 libwayland-egl1 libwayland-server0"
DNF_PKGS="curl ca-certificates glib2 dbus-libs fontconfig freetype mesa-libEGL mesa-libGL libdrm
  libX11 libX11-xcb libxkbcommon libxkbcommon-x11 libxcb xcb-util xcb-util-cursor xcb-util-image
  xcb-util-keysyms xcb-util-renderutil xcb-util-wm libwayland-client libwayland-cursor
  libwayland-egl libwayland-server"
system_packages() {
  local mgr install missing="" pkg alt
  if command -v apt-get >/dev/null; then
    mgr=apt; set -- $APT_PKGS
    have() { dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "ok installed"; }
    known() { apt-cache show "$1" >/dev/null 2>&1; }
    install="apt-get install -y"
  elif command -v dnf >/dev/null; then
    mgr=dnf; set -- $DNF_PKGS
    have() { rpm -q "$1" >/dev/null 2>&1; }
    known() { dnf -q list --available "$1" >/dev/null 2>&1 || rpm -q "$1" >/dev/null 2>&1; }
    install="dnf install -y --setopt=install_weak_deps=False"
  else
    echo "Note: unknown package manager; if Freeroom won't start, install curl and Qt's X11/Wayland libraries." >&2
    return 0
  fi
  for pkg in "$@"; do
    local found=""
    for alt in ${pkg//|/ }; do have "$alt" && found=1 && break; done
    [ -n "$found" ] && continue
    for alt in ${pkg//|/ }; do known "$alt" && missing="$missing $alt" && break; done
  done
  [ -z "$missing" ] && return 0
  echo "Installing system packages ($mgr):$missing"
  local SUDO=""
  [ "$(id -u)" -ne 0 ] && SUDO=sudo
  [ "$mgr" = apt ] && $SUDO apt-get update -qq
  # shellcheck disable=SC2086
  $SUDO $install $missing
}
# After the install: any library Qt (or OpenCV / rawpy) still can't find?
missing_libs() {
  local sp="$HERE/.venv/lib/python3.12/site-packages" q f
  q="$sp/PySide6/Qt"
  for f in "$q"/plugins/platforms/*.so "$q"/plugins/wayland-*/*.so "$q"/plugins/xcbglintegrations/*.so \
           "$q"/lib/libQt6{Core,Gui,Widgets,DBus,OpenGL,XcbQpa,WaylandClient,WaylandEglClientHwIntegration}.so.6 \
           "$sp"/PySide6/Qt{Core,Gui,Widgets}.*.so "$sp"/cv2/cv2*.so "$sp"/rawpy/*.so; do
    [ -f "$f" ] || continue
    ldd "$f" 2>/dev/null | grep "not found" | sed 's/ =>.*//; s/^[[:space:]]*//' || true
  done | sort -u
}
if [ "$MODE" = native ] && [ "$(uname -s)" = Linux ]; then
  system_packages
fi

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
  # shellcheck disable=SC2088  # expanded by the shell that starts the app
  UV='~/.local/bin/uv'  # inside the box
else
  echo "Installing uv + Python deps…"
  bash -lc "$SYNC"
  # wherever uv is (~/.local/bin, Homebrew…): app launchers get a minimal PATH
  UV="$(bash -lc 'export PATH=$HOME/.local/bin:$PATH; command -v uv')"
fi

printf 'mode=%s\nbox=%s\nextra=%s\nuv=%s\n' "$MODE" "$BOX" "$EXTRA" "$UV" > "$HERE/.install"

if [ "$MODE" = native ] && [ "$(uname -s)" = Linux ]; then
  LEFT="$(missing_libs)"
  if [ -n "$LEFT" ]; then
    echo "Warning: these libraries are still missing, Freeroom may not start:" >&2
    echo "$LEFT" | sed 's/^/  /' >&2
    echo "Install the packages that provide them with your package manager." >&2
  fi
fi

"$HERE/setup/download-models.sh"
"$HERE/setup/install-launcher.sh"

echo
# shellcheck disable=SC2088  # just text
if [ "$(uname -s)" = Darwin ]; then WHERE="~/Applications (Freeroom)"; else WHERE="the app menu (Freeroom)"; fi
echo "Done. Start it from $WHERE or: $HERE/freeroom.sh [photo]"
