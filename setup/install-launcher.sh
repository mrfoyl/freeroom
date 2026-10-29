#!/usr/bin/env bash
# Adds "Freeroom" to the app menu; freeroom.sh starts it the way it was installed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
# Own icon if present (kept out of git), otherwise a generic one
ICON="$HERE/freeroom/assets/freeroom.png"
[ -f "$ICON" ] || ICON=image-x-generic
cat > "$APPS/freeroom.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Freeroom
Comment=RAW development and AI photo noise reduction
Exec=$HERE/freeroom.sh %f
Icon=$ICON
Terminal=false
Categories=Graphics;Photography;
MimeType=image/jpeg;image/png;image/tiff;image/webp;image/x-canon-cr2;image/x-canon-cr3;image/x-nikon-nef;image/x-sony-arw;image/x-fuji-raf;image/x-adobe-dng;
StartupWMClass=freeroom
DESK
update-desktop-database "$APPS" 2>/dev/null || true
echo "Installed $APPS/freeroom.desktop"
