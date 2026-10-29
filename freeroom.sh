#!/usr/bin/env bash
# Launch the Freeroom GUI (in its distrobox, or natively; see setup/install.sh).
# Optional arg: a photo to open.
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE=box
BOX="${FREEROOM_BOX:-freeroom}"
UV='~/.local/bin/uv'
if [ -f "$HERE/.install" ]; then
  MODE="$(sed -n 's/^mode=//p' "$HERE/.install")"
  BOX="${FREEROOM_BOX:-$(sed -n 's/^box=//p' "$HERE/.install")}"
  UV="$(sed -n 's/^uv=//p' "$HERE/.install")"
  UV="${UV:-~/.local/bin/uv}"
fi
# --no-sync: a plain sync would drop the PyTorch build (it's an extra)
RUN="cd '$HERE' && exec $UV run --no-sync freeroom \"\$@\""
if [ "$MODE" = native ]; then
  exec bash -c "$RUN" _ "$@"
fi
exec distrobox enter "$BOX" -- bash -lc "$RUN" _ "$@"
