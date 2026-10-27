#!/usr/bin/env bash
# Launch the Freeroom GUI inside its distrobox. Optional arg: a photo to open.
HERE="$(cd "$(dirname "$0")" && pwd)"
BOX="${FREEROOM_BOX:-freeroom}"
exec distrobox enter "$BOX" -- bash -lc "cd '$HERE' && exec ~/.local/bin/uv run --no-sync freeroom \"\$@\"" _ "$@"
