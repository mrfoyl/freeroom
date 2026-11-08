#!/usr/bin/env bash
# Downloads the models:
#   models/denoise/  SCUNet real-world denoising (KAIR, Apache-2.0)
#   models/sharpen/  Restormer defocus + motion deblurring (swz30/Restormer)
#   models/segment/  SAM 2.1 small, AI selection for layers (Meta, Apache-2.0)
# Any other 1x RGB model spandrel can load can be dropped into either folder
# and shows up in that menu.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$DIR/denoise" "$DIR/sharpen" "$DIR/segment/sam2.1-hiera-small"
# Older layout kept the denoise models directly in models/
for f in "$DIR"/*.pth; do [ -e "$f" ] && mv -n "$f" "$DIR/denoise/"; done

get() {  # url dest
  if [ ! -s "$2" ]; then
    echo "Downloading $(basename "$2")…"
    curl -fL --retry 3 -o "$2.part" "$1"
    mv "$2.part" "$2"
  fi
}
KAIR=https://github.com/cszn/KAIR/releases/download/v1.0
for f in scunet_color_real_psnr scunet_color_real_gan; do
  get "$KAIR/$f.pth" "$DIR/denoise/$f.pth"
done
RESTORMER=https://github.com/swz30/Restormer/releases/download/v1.0
for f in single_image_defocus_deblurring motion_deblurring; do
  get "$RESTORMER/$f.pth" "$DIR/sharpen/restormer_$f.pth"
done
SAM=https://huggingface.co/facebook/sam2.1-hiera-small/resolve/main
for f in config.json preprocessor_config.json processor_config.json model.safetensors; do
  get "$SAM/$f" "$DIR/segment/sam2.1-hiera-small/$f"
done
ls -lh "$DIR"/*/
