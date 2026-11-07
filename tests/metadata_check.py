"""Metadata survives saving: camera, lens, exposure, date, GPS, XMP and colour
profile from the source go into saved JPEG, PNG and TIFF files.

    python tests/metadata_check.py photo [photo …]   (RAW or JPEG; only read)

Saved files go to tests/out/meta and are deleted afterwards.
"""
import datetime, hashlib, shutil, struct, sys
from pathlib import Path

import imagecodecs
import numpy as np
import tifffile
from PIL import Image

from freeroom import imageio, metadata

out_dir = Path(__file__).parent / "out" / "meta"
out_dir.mkdir(parents=True, exist_ok=True)
fails = []
KEYS = ("camera", "lens", "iso", "shutter", "aperture", "focal", "taken")


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name, extra)
    if not ok:
        fails.append(name)


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gps_of(p: Path) -> dict:
    ex, _ = metadata.read_exif(p)
    return dict(ex.get_ifd(metadata.GPS_IFD)) if ex is not None else {}


for src in map(Path, sys.argv[1:]):
    before = digest(src)
    photo = imageio.load(src)
    want = {k: photo.meta["shot"].get(k) for k in KEYS}
    src_gps = gps_of(src)
    print(f"\n{src.name}: {want}  GPS tags {len(src_gps)}  XMP {bool(photo.xmp)}")
    rng = np.random.default_rng(0)
    rgb = rng.random((300, 400, 3)).astype(np.float32)  # pixels don't matter here
    for ext in (".jpg", ".png", ".tif"):
        out = out_dir / (src.stem + "_" + src.suffix[1:] + ext)
        imageio.save(photo, rgb, out)
        got = imageio.shot_info(out)
        same = {k: got.get(k) for k in KEYS}
        diff = {k: (want[k], same[k]) for k in KEYS if want[k] is not None and not (
            same[k] == want[k] or isinstance(want[k], float) and same[k] and abs(same[k] - want[k]) < 1e-3 * want[k])}
        check(f"{src.suffix} -> {ext}: camera, lens, exposure, date kept", not diff, str(diff) if diff else "")
        ex, xmp = metadata.read_exif(out)
        sub = ex.get_ifd(metadata.EXIF_IFD) if ex is not None else {}
        check(f"{src.suffix} -> {ext}: upright, saved size, Software",
              ex is not None and ex.get(274) == 1 and sub.get(40962) == 400 and sub.get(40963) == 300
              and str(ex.get(305, "")).startswith("Freeroom"),
              f"orientation {ex and ex.get(274)} size {sub.get(40962)}x{sub.get(40963)} software {ex and ex.get(305)}")
        check(f"{src.suffix} -> {ext}: GPS kept", gps_of(out) == src_gps if src_gps else True,
              f"{len(gps_of(out))} of {len(src_gps)} tags")
        if photo.xmp:
            check(f"{src.suffix} -> {ext}: XMP kept", bool(xmp) and b"tiff:Orientation" not in xmp)
        no_maker = 37500 not in sub
        check(f"{src.suffix} -> {ext}: no maker note copied", no_maker)
        if ext == ".tif":
            px = tifffile.imread(out)
            check(".tif pixels 16-bit and intact", px.dtype == np.uint16 and px.shape == (300, 400, 3)
                  and np.abs(px / 65535 - rgb).max() < 1e-4)
        elif ext == ".png":
            with Image.open(out) as im:
                im.load()
                has_icc = bool(im.info.get("icc_profile"))
            px = __import__("cv2").imread(str(out), -1)
            check(".png pixels 16-bit, ICC present", px.dtype == np.uint16 and px.shape == (300, 400, 3) and has_icc)
        else:
            with Image.open(out) as im:
                check(".jpg ICC present", bool(im.info.get("icc_profile")))
    for ext in (".tif", ".png"):  # 8-bit: smaller, same metadata
        out = out_dir / (src.stem + "_8bit" + ext)
        imageio.save(photo, rgb, out, bits=8)
        px = tifffile.imread(out) if ext == ".tif" else __import__("cv2").imread(str(out), -1)[..., ::-1]
        got = imageio.shot_info(out)
        check(f"{src.suffix} -> 8-bit {ext}: 8-bit pixels intact, metadata kept",
              px.dtype == np.uint8 and np.abs(px / 255 - rgb).max() <= 0.5 / 255 + 1e-6
              and all(got.get(k) == want[k] or isinstance(want[k], float) and got.get(k) and abs(got[k] - want[k]) < 1e-3 * want[k]
                      for k in KEYS if want[k] is not None))
    for name, dist in (("jxl_std", 1.0), ("jxl_hq", 0.5), ("jxl_lossless", 0.0)):  # JPEG XL: 16-bit + boxes
        out = out_dir / f"{src.stem}_{name}.jxl"
        imageio.save(photo, rgb, out, jxl_distance=dist)
        px = imagecodecs.jpegxl_decode(out.read_bytes())
        err = np.abs(px / 65535 - rgb).mean() * 255
        boxes, data, i = {}, out.read_bytes(), 0
        while i < len(data):
            size, kind = struct.unpack_from(">I4s", data, i)
            boxes[kind] = data[i + 8 : i + size]
            i += size
        ex = Image.Exif()
        if b"Exif" in boxes:
            ex.load(b"Exif\0\0" + boxes[b"Exif"][4:])
        sub = ex.get_ifd(metadata.EXIF_IFD)
        model = str(ex.get(272, "")).strip("\0 ")
        check(f"{src.suffix} -> .jxl {name}: 16-bit, metadata boxes",
              px.dtype == np.uint16 and px.shape == (300, 400, 3) and (err < 1e-3 if dist == 0 else err < 20)
              and (not want["camera"] or model and model in want["camera"]) and sub.get(40962) == 400
              and (b"xml " in boxes) == bool(photo.xmp) and (not src_gps or len(ex.get_ifd(metadata.GPS_IFD)) == len(src_gps)),
              f"mean error {err:.3f}/255, model {model!r}, {out.stat().st_size / 1e3:.0f} kB")
    check(f"{src.name} untouched", digest(src) == before)

# RAWs Pillow can't read (CR3, RAF…): the basics from LibRaw are written
blank = out_dir / "no_exif.png"
Image.new("RGB", (8, 8)).save(blank)
shot = {"camera": "Canon EOS R5", "lens": "RF24-105mm F4 L IS USM", "iso": 400.0, "shutter": 1 / 250,
        "aperture": 5.6, "focal": 50.0, "taken": datetime.datetime(2026, 1, 2, 3, 4, 5)}
exif, _ = metadata.carry(blank, shot)
photo = imageio.Photo(blank, np.zeros((8, 8, 3), np.float32), exif=exif)
imageio.save(photo, np.zeros((30, 40, 3), np.float32), out_dir / "fallback.jpg")
got = imageio.shot_info(out_dir / "fallback.jpg")
check("no readable EXIF: LibRaw basics written",
      all(got.get(k) == v or isinstance(v, float) and abs(got.get(k, 0) - v) < 1e-3 * v for k, v in shot.items()),
      str({k: got.get(k) for k in shot}))

shutil.rmtree(out_dir)
print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
