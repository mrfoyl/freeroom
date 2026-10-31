"""Loading photos as scene-linear float32 RGB, saving developed sRGB.

Source files are only ever opened for reading. Saving refuses to write over
the file the photo was loaded from.
"""

from __future__ import annotations

import datetime
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
    photo = _load_raw(path) if ext in RAW_EXTS else _load_raster(path)
    photo.meta["shot"] = shot_info(path)
    return photo


# --- shooting metadata -------------------------------------------------------
EXPOSURE_PROGRAMS = {1: "Manual", 2: "Program", 3: "Aperture priority", 4: "Shutter priority",
                     5: "Creative", 6: "Action", 7: "Portrait", 8: "Landscape"}


def _num(v) -> float | None:
    """EXIF number (int, float, IFDRational, (num, den) or a 1-tuple) -> float."""
    try:
        if isinstance(v, (tuple, list)):
            v = v[0] / v[1] if len(v) == 2 else v[0]
        f = float(v)
        return f if f == f else None  # NaN (0/0 rationals)
    except (TypeError, ValueError, ZeroDivisionError, IndexError):
        return None


def _text(v) -> str:
    if isinstance(v, bytes):
        v = v.decode("utf-8", "ignore")
    return str(v or "").strip("\x00 ").strip()


def shot_info(path: str | Path) -> dict:
    """Camera, lens and exposure of a photo, from EXIF (Pillow; also reads
    TIFF-based RAWs such as CR2, NEF, ARW, DNG) and, for RAWs, LibRaw.
    Missing fields are left out. Only reads the file."""
    path = Path(path)
    info: dict = {}
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769)
            make, model = _text(ex.get(271)), _text(ex.get(272))
            info["camera"] = model if model.lower().startswith(make.lower()) else f"{make} {model}".strip()
            lens_make, lens = _text(sub.get(42035)), _text(sub.get(42036))
            info["lens"] = lens if not lens_make or lens.lower().startswith(lens_make.lower()) else f"{lens_make} {lens}"
            iso = sub.get(34855)
            info["iso"] = _num(iso[0] if isinstance(iso, (tuple, list)) else iso)
            info["shutter"] = _num(sub.get(33434))
            info["aperture"] = _num(sub.get(33437))
            info["focal"] = _num(sub.get(37386))
            info["focal35"] = _num(sub.get(41989))
            info["bias"] = _num(sub.get(37380))
            if (flash := sub.get(37385)) is not None:
                info["flash"] = bool(int(flash) & 1)
            info["program"] = EXPOSURE_PROGRAMS.get(sub.get(34850))
            if taken := _text(sub.get(36867) or ex.get(306)):
                try:
                    info["taken"] = datetime.datetime.strptime(taken[:19], "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass
    except Exception:
        pass  # not readable by Pillow (CR3, RAF, …): LibRaw below
    if path.suffix.lower() in RAW_EXTS:
        try:
            import rawpy

            with rawpy.imread(str(path)) as raw:
                o, lens = raw.other, raw.lens
            for key, val in (("iso", o.iso_speed), ("shutter", o.shutter_speed),
                             ("aperture", o.aperture), ("focal", o.focal_length)):
                if not info.get(key) and val and val > 0:
                    info[key] = float(val)
            if not info.get("lens") and lens.model:
                info["lens"] = _text(lens.model)
            if not info.get("taken") and o.timestamp and o.timestamp.year > 1971:
                info["taken"] = o.timestamp
        except Exception:
            pass
    # drop unknowns; 0 means "not recorded" for these (but not for bias or flash)
    unset = {"iso", "shutter", "aperture", "focal", "focal35"}
    return {k: v for k, v in info.items() if v is not None and v != "" and not (k in unset and not v)}


def describe_shot(info: dict) -> tuple[str, list[tuple[str, str]]]:
    """("ISO 100 · 1/125 s · f/7.1 · 55 mm", [(label, value), …]) for display."""
    head = []
    if iso := info.get("iso"):
        head.append(f"ISO {iso:g}")
    if t := info.get("shutter"):
        head.append(f"{round(t, 1):g} s" if t >= 0.3 else f"1/{round(1 / t)} s")
    if a := info.get("aperture"):
        head.append(f"f/{round(a, 1):g}")
    if f := info.get("focal"):
        f35 = info.get("focal35")
        head.append(f"{f:g} mm" + (f" ({f35:g} mm FF)" if f35 and abs(f35 - f) > 1 else ""))
    rows = []
    if (b := info.get("bias")) is not None:
        rows.append(("Exp. comp.", f"{b:+.1f} EV" if abs(b) >= 0.05 else "0 EV"))
    if p := info.get("program"):
        rows.append(("Mode", p))
    if "flash" in info:
        rows.append(("Flash", "Fired" if info["flash"] else "No"))
    if c := info.get("camera"):
        rows.append(("Camera", c))
    if lens := info.get("lens"):
        rows.append(("Lens", lens))
    if d := info.get("taken"):
        rows.append(("Taken", d.strftime("%Y-%m-%d %H:%M:%S")))
    return " · ".join(head), rows


def _load_raw(path: Path) -> Photo:
    import rawpy

    try:
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
    except rawpy.LibRawFileUnsupportedError:
        # e.g. DNG 1.7 with JPEG XL tiles (recent Samsung phones), which LibRaw can't decode
        if path.suffix.lower() == ".dng":
            return _load_linear_dng(path)
        raise
    linear = rgb16.astype(np.float32) / 65535.0
    return Photo(path, linear, baseline_ev=auto_exposure(linear), meta={"bits": 16, "raw": True})


# XYZ (D50) -> linear sRGB (D65), Bradford adapted
XYZ_D50_TO_SRGB = np.array([[3.1338561, -1.6168667, -0.4906146],
                            [-0.9787684, 1.9161415, 0.0334540],
                            [0.0719453, -0.2289914, 1.4052427]])
# EXIF LightSource codes used as DNG calibration illuminants -> colour temperature (K)
ILLUMINANT_K = {1: 5500, 2: 4200, 3: 2850, 4: 5500, 9: 5500, 10: 6500, 11: 7500, 12: 6400,
                13: 5000, 14: 4200, 15: 3450, 17: 2856, 18: 4874, 19: 6774, 20: 5503,
                21: 6504, 22: 7504, 23: 5003, 24: 3200}


def _load_linear_dng(path: Path) -> Photo:
    """Already-demosaiced (LinearRaw) DNG via tifffile + imagecodecs, colour per the DNG spec."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        top = tif.pages[0]
        page = next((p for p in [top, *(top.pages or [])] if p.photometric == 34892), None)
        if page is None:
            raise ValueError(f"Unsupported DNG (no LinearRaw image): {path}")
        tags = {t.code: t.value for t in top.tags.values()} | {t.code: t.value for t in page.tags.values()}
        data = page.asarray().astype(np.float32)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Unsupported DNG ({data.shape} LinearRaw): {path}")

    def mat(code, n=3):
        v = tags.get(code)
        return None if v is None else (np.array(v[0::2], float) / np.array(v[1::2], float)).reshape(n, -1)

    black = np.array(tags.get(50714, (0,)), float)
    black = black.reshape(-1, 3).mean(0) if black.size % 3 == 0 else black.mean()
    white = np.array(tags.get(50717, (65535,)), float)
    cam = (data - black) / (white - black)
    if (crop := tags.get(50829)) and len(crop) == 4:  # ActiveArea top, left, bottom, right
        cam = cam[crop[0]:crop[2], crop[1]:crop[3]]

    neutral = mat(50728, 1).ravel() if 50728 in tags else np.ones(3)
    cm = [mat(50721), mat(50722)]
    fm = [mat(50964), mat(50965)]
    cc = [mat(50723) if 50723 in tags else np.eye(3), mat(50724) if 50724 in tags else np.eye(3)]
    temps = [ILLUMINANT_K.get(tags.get(50778), 6504), ILLUMINANT_K.get(tags.get(50779), 2856)]

    def blend(pair, g):
        return pair[0] if pair[1] is None else g * pair[0] + (1 - g) * pair[1]

    # Find the white point (and so the interpolation weight) from AsShotNeutral, iterating.
    g, xy = 1.0, np.array([0.3457, 0.3585])
    for _ in range(20):
        x, y = xy
        n = (x - 0.3320) / (0.1858 - y)  # McCamy CCT approximation
        cct = np.clip(449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33, 2000, 12000)
        t1, t2 = temps
        g = 1.0 if t1 == t2 else float(np.clip((1 / cct - 1 / t2) / (1 / t1 - 1 / t2), 0, 1))
        xyz = np.linalg.solve(blend(cc, g) @ blend(cm, g), neutral)
        xy = xyz[:2] / xyz.sum()

    cal = blend(cc, g)
    if fm[0] is not None:
        wb = np.linalg.solve(cal, neutral)
        cam_to_xyz = blend(fm, g) @ np.diag(1 / wb) @ np.linalg.inv(cal)
        to_srgb = XYZ_D50_TO_SRGB @ cam_to_xyz
    else:
        # No ForwardMatrix: white-balance on the neutral, then the inverse ColorMatrix (D65-ish)
        to_srgb = np.linalg.inv(blend(cm, g) @ cal) @ np.diag(1 / neutral)
        to_srgb /= (to_srgb @ neutral).max()
    linear = np.maximum(cam.reshape(-1, 3) @ to_srgb.T.astype(np.float32), 0).reshape(cam.shape)

    rot = {3: cv2.ROTATE_180, 6: cv2.ROTATE_90_CLOCKWISE, 8: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(tags.get(EXIF_ORIENTATION))
    if rot is not None:
        linear = cv2.rotate(linear, rot)
    linear = np.ascontiguousarray(linear, dtype=np.float32)
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
