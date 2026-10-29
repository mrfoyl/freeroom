#!/usr/bin/env bash
# Adds "Freeroom" to the app menu (Linux) or ~/Applications (macOS);
# freeroom.sh starts it the way it was installed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# Own icon if present (kept out of git), otherwise a generic one
PNG="$HERE/freeroom/assets/freeroom.png"

if [ "$(uname -s)" = Darwin ]; then
  # A minimal app bundle whose executable runs freeroom.sh. Photos open by
  # dropping them on the window (a script app doesn't get Finder's "Open with").
  APP="$HOME/Applications/Freeroom.app"
  mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
  cat > "$APP/Contents/MacOS/Freeroom" <<RUN
#!/bin/bash
exec "$HERE/freeroom.sh" "\$@"
RUN
  chmod +x "$APP/Contents/MacOS/Freeroom"
  ICON_KEY=""
  if [ -f "$PNG" ]; then
    SET="$(mktemp -d)/freeroom.iconset"
    mkdir -p "$SET"
    for size in 16 32 128 256 512; do
      sips -z $size $size "$PNG" --out "$SET/icon_${size}x${size}.png" >/dev/null
      sips -z $((size * 2)) $((size * 2)) "$PNG" --out "$SET/icon_${size}x${size}@2x.png" >/dev/null
    done
    iconutil -c icns "$SET" -o "$APP/Contents/Resources/freeroom.icns"
    ICON_KEY="<key>CFBundleIconFile</key><string>freeroom</string>"
  fi
  cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Freeroom</string>
  <key>CFBundleDisplayName</key><string>Freeroom</string>
  <key>CFBundleIdentifier</key><string>local.freeroom</string>
  <key>CFBundleExecutable</key><string>Freeroom</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>  <!-- PySide6 6.12 wheels -->
  <key>NSHighResolutionCapable</key><true/>
  $ICON_KEY
</dict>
</plist>
PLIST
  touch "$APP"  # make Finder pick up the new bundle
  echo "Installed $APP"
  exit 0
fi

APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
ICON="$PNG"
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
