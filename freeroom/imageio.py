"""Loading photos as scene-linear float32 RGB, saving developed sRGB.

Source files are only ever opened for reading. Saving refuses to write over
the file the photo was loaded from.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .develop import auto_exposure, srgb_to_linear

RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf", ".dng", ".pef", ".srw"}
RASTER_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
ALL_EXTS = RAW_EXTS | RASTER_EXTS

EXIF_ORIENTATION = 0x0112


@dataclass
class Photo:
    path: Path
    linear: np.ndarray  # HxWx3 float32 scene-linear RGB (sRGB primaries), upright
    baseline_ev: float = 0.0  # exposure that makes a RAW look normal; 0 for JPEG etc.
    exif: bytes | None = None
    icc: bytes | None = None
    meta: dict = field(default_factory=dict)


def load(path: str | Path) -> Photo:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in RAW_EXTS:
        return _load_raw(path)
    return _load_raster(path)


def _load_raw(path: Path) -> Photo:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            output_bps=16,
            gamma=(1, 1),  # linear: development does the tone curve
            no_auto_bright=True,  # keep sensor white at 1.0; auto_exposure sets the baseline
            highlight_mode=rawpy.HighlightMode.Blend,
            output_color=rawpy.ColorSpace.sRGB,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    linear = rgb16.astype(np.float32) / 65535.0
    return Photo(path, linear, baseline_ev=auto_exposure(linear), meta={"bits": 16, "raw": True})


def _load_raster(path: Path) -> Photo:
    # cv2 applies EXIF orientation by default, so the array is upright.
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.dtype in (np.uint8, np.uint16):
        # Linearise through a lookup table (much faster than pow on every pixel).
        bits = 8 if rgb.dtype == np.uint8 else 16
        table = srgb_to_linear(np.arange(2**bits, dtype=np.float32) / (2**bits - 1))
        linear = table[rgb]
    else:
        bits = 32
        linear = srgb_to_linear(rgb)

    exif = icc = None
    try:
        with Image.open(path) as im:
            icc = im.info.get("icc_profile")
            ex = im.getexif()
            if ex:
                # Pixels are already rotated upright, so reset the tag.
                ex[EXIF_ORIENTATION] = 1
                exif = ex.tobytes()
    except Exception:
        pass
    return Photo(path, linear, exif=exif, icc=icc, meta={"bits": bits, "raw": False})


def save(photo: Photo, rgb: np.ndarray, out_path: str | Path, jpeg_quality: int = 95) -> Path:
    """Save developed display-sRGB `rgb` (0..1)."""
    out_path = Path(out_path)
    if out_path.resolve() == photo.path.resolve():
        raise ValueError("Refusing to overwrite the original photo. Pick another file name.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    rgb = np.clip(rgb, 0.0, 1.0)

    if ext in {".tif", ".tiff", ".png"}:
        # 16-bit lossless
        bgr16 = cv2.cvtColor((rgb * 65535.0 + 0.5).astype(np.uint16), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(out_path), bgr16):
            raise OSError(f"Could not write {out_path}")
        return out_path

    im = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), "RGB")
    kwargs = {}
    if photo.exif:
        kwargs["exif"] = photo.exif
    if photo.icc:
        kwargs["icc_profile"] = photo.icc
    if ext in {".jpg", ".jpeg"}:
        kwargs.update(quality=jpeg_quality, subsampling=0)
    buf = io.BytesIO()
    im.save(buf, format=Image.registered_extensions().get(ext, "JPEG"), **kwargs)
    out_path.write_bytes(buf.getvalue())
    return out_path
