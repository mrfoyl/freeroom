"""Photo metadata carried from the source file to the saved one.

Reads EXIF (camera, lens, exposure, date, GPS, copyright…) and XMP from the
source, including TIFF-based RAWs (CR2, NEF, ARW, DNG, and ORF / RW2, which
only differ in their header), and writes it into saved JPEG, PNG and TIFF
files. RAWs Pillow can't read (CR3, RAF…) get the basics LibRaw reports.

Left out on purpose: the maker note (its offsets break when it moves), the
thumbnail, and anything describing the old pixels (orientation is 1 because
the saved pixels are upright, sizes are the saved size).
"""

from __future__ import annotations

import io
import re
import struct
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms
from PIL.TiffImagePlugin import IFDRational

EXIF_IFD, GPS_IFD = 0x8769, 0x8825
# IFD0 tags worth keeping: description, make, model, software, date, artist, copyright
IFD0_KEEP = (270, 271, 272, 305, 306, 315, 33432)
# Exif IFD tags dropped: maker note, interoperability pointer, old pixel size
EXIF_DROP = {37500, 40965, 40962, 40963}
# TIFF-based RAWs whose header Pillow doesn't recognise
TIFF_HEADERS = {b"IIRO": b"II*\0", b"IIRS": b"II*\0", b"MMOR": b"MM\0*", b"IIU\0": b"II*\0"}
_XMP_ORIENTATION = re.compile(rb'\s*tiff:Orientation\s*=\s*"[^"]*"|<tiff:Orientation>[^<]*</tiff:Orientation>')


def read_exif(path: str | Path) -> tuple[Image.Exif | None, bytes | None]:
    """(EXIF, XMP) of a file, either None when missing. Only reads the file."""
    path = Path(path)
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            xmp = im.info.get("xmp") or ex.get(700)
            ex.get_ifd(EXIF_IFD), ex.get_ifd(GPS_IFD)  # read them while the file is open
            return ex, _bytes(xmp)
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            head = f.read(4)
            if head not in TIFF_HEADERS:
                return None, None
            data = TIFF_HEADERS[head] + f.read()
        ex = Image.Exif()
        ex.load(data)
        ex.get_ifd(EXIF_IFD), ex.get_ifd(GPS_IFD)
        return ex, _bytes(ex.get(700))
    except Exception:
        return None, None


def _bytes(xmp) -> bytes | None:
    if isinstance(xmp, str):
        xmp = xmp.encode("utf-8")
    if isinstance(xmp, tuple):  # some TIFFs store it as a tuple of byte values
        xmp = bytes(xmp)
    return xmp if isinstance(xmp, (bytes, bytearray)) and xmp.strip() else None


def _rational(x: float) -> IFDRational:
    f = Fraction(float(x)).limit_denominator(100_000)
    return IFDRational(f.numerator, f.denominator)


def carry(path: str | Path, shot: dict | None = None) -> tuple[bytes | None, bytes | None]:
    """EXIF (as bytes for saving) and XMP to carry over from `path`.
    `shot` (imageio.shot_info) fills in the basics when the file has no
    readable EXIF, as for CR3 or RAF."""
    src, xmp = read_exif(path)
    out = Image.Exif()
    sub: dict = {}
    gps: dict = {}
    if src is not None:
        out.endian = getattr(src, "endian", None)
        for tag in IFD0_KEEP:
            if src.get(tag) not in (None, "", b""):
                out[tag] = src[tag]
        sub = {k: v for k, v in src.get_ifd(EXIF_IFD).items() if k not in EXIF_DROP}
        gps = dict(src.get_ifd(GPS_IFD))
    if shot and not sub:
        if cam := shot.get("camera"):
            out[272] = cam
        if iso := shot.get("iso"):
            sub[34855] = int(iso)
        for tag, key in ((33434, "shutter"), (33437, "aperture"), (37386, "focal")):
            if v := shot.get(key):
                sub[tag] = _rational(v)
        if lens := shot.get("lens"):
            sub[42036] = lens
        if taken := shot.get("taken"):
            sub[36867] = taken.strftime("%Y:%m:%d %H:%M:%S")
    out[305] = "Freeroom"
    out[274] = 1  # the saved pixels are upright
    if sub:
        out[EXIF_IFD] = sub
    if gps:
        out[GPS_IFD] = gps
    if xmp:
        xmp = _XMP_ORIENTATION.sub(b"", xmp)
    return _serialise(out), xmp


def _serialise(ex: Image.Exif) -> bytes | None:
    """EXIF bytes; tags Pillow can't write back are dropped one by one."""
    try:
        return ex.tobytes()
    except Exception:
        pass
    for ifd in (EXIF_IFD, GPS_IFD):
        if isinstance(ex.get(ifd), dict):
            ok = {}
            for k, v in ex[ifd].items():
                test = Image.Exif()
                test[ifd] = {k: v}
                try:
                    test.tobytes()
                    ok[k] = v
                except Exception:
                    pass
            ex[ifd] = ok
    for k in list(ex.keys()):
        if k in (EXIF_IFD, GPS_IFD):
            continue
        test = Image.Exif()
        test[k] = ex[k]
        try:
            test.tobytes()
        except Exception:
            del ex[k]
    try:
        return ex.tobytes()
    except Exception:
        return None


def for_output(exif: bytes | None, width: int, height: int) -> Image.Exif | None:
    """The carried EXIF with the saved photo's pixel size."""
    if not exif:
        return None
    ex = Image.Exif()
    ex.load(exif)
    sub = ex.get_ifd(EXIF_IFD)
    sub[40962], sub[40963] = width, height
    ex[EXIF_IFD] = dict(sub)
    gps = ex.get_ifd(GPS_IFD)
    if gps:
        ex[GPS_IFD] = dict(gps)
    return ex


def srgb_icc() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


# --- PNG: eXIf, iCCP and XMP chunks after IHDR ---------------------------------
def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def add_png_metadata(png: bytes, ex: Image.Exif | None, xmp: bytes | None, icc: bytes | None) -> bytes:
    chunks = b""
    if icc:
        chunks += _chunk(b"iCCP", b"ICC profile\0\0" + zlib.compress(icc))
    if ex is not None:
        chunks += _chunk(b"eXIf", ex.tobytes()[6:])  # without the "Exif\0\0" JPEG prefix
    if xmp:
        chunks += _chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0" + xmp)
    end_ihdr = 8 + 8 + 13 + 4  # signature, IHDR length+type, IHDR data, CRC
    return png[:end_ihdr] + chunks + png[end_ihdr:]


# --- JPEG XL: the codestream in a container with Exif and XMP boxes ----------
def _box(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", 8 + len(data)) + kind + data


def jxl_container(encoded: bytes, ex: Image.Exif | None, xmp: bytes | None) -> bytes:
    """`encoded` is a bare codestream, or already a container (the encoder makes
    one when it needs a level box, as for 16-bit lossless): its boxes are kept."""
    signature = _box(b"JXL ", b"\r\n\x87\n")
    if encoded.startswith(signature):
        level, rest, i = b"", b"", 0
        while i < len(encoded):
            size, kind = struct.unpack_from(">I4s", encoded, i)
            size = size or len(encoded) - i  # 0 = up to the end
            if kind == b"jxll":
                level = encoded[i : i + size]
            elif kind not in (b"JXL ", b"ftyp"):
                rest += encoded[i : i + size]
            i += size
    else:
        level, rest = b"", _box(b"jxlc", encoded)
    out = signature + _box(b"ftyp", b"jxl \0\0\0\0jxl ") + level  # the level box must come first
    if ex is not None:
        out += _box(b"Exif", b"\0\0\0\0" + ex.tobytes()[6:])  # offset 0 to the TIFF header
    if xmp:
        out += _box(b"xml ", xmp)
    return out + rest


# --- TIFF: tifffile for the 8- or 16-bit pixels, the Exif/GPS IFDs appended ---------
def write_tiff(path: Path, pixels: np.ndarray, ex: Image.Exif | None, xmp: bytes | None, icc: bytes | None):
    import tifffile

    tags = [(274, "H", 1, 1, True)]
    kwargs: dict = {"software": "Freeroom", "metadata": None}
    sub_ifds = {}
    if ex is not None:
        for tag in (271, 272, 315, 33432):
            if isinstance(v := ex.get(tag), str) and v.strip("\0 "):
                tags.append((tag, "s", 0, v.strip("\0"), True))
        if isinstance(d := ex.get(270), str) and d.strip("\0 "):
            kwargs["description"] = d.strip("\0")
        if isinstance(t := ex.get(306), str) and t.strip("\0 "):
            kwargs["datetime"] = t.strip("\0")[:19]
        sub_ifds = {k: ex.get_ifd(k) for k in (EXIF_IFD, GPS_IFD) if ex.get_ifd(k)}
    if icc:
        tags.append((34675, "B", len(icc), icc, True))
    if xmp:
        tags.append((700, "B", len(xmp), xmp, True))

    def tiff(pointers: dict) -> bytes:
        # tifffile won't write the Exif/GPS pointer tags, so it writes stand-ins one
        # below them (same place in the sorted tag list), renamed afterwards.
        buf = io.BytesIO()
        # Deflate with the horizontal predictor: lossless and about half the size of
        # LZW, which grows noisy 16-bit RAW data beyond its uncompressed size.
        tifffile.imwrite(buf, pixels, photometric="rgb", compression="zlib", predictor=True, byteorder="<",
                         extratags=tags + [(k - 1, "I", 1, v, True) for k, v in pointers.items()], **kwargs)
        data = bytearray(buf.getvalue())
        ifd0 = struct.unpack_from("<I", data, 4)[0]
        for i in range(struct.unpack_from("<H", data, ifd0)[0]):
            at = ifd0 + 2 + 12 * i
            if struct.unpack_from("<H", data, at)[0] + 1 in pointers:
                struct.pack_into("<H", data, at, struct.unpack_from("<H", data, at)[0] + 1)
        return bytes(data)

    data = tiff({k: 0 for k in sub_ifds})
    if sub_ifds:
        start = len(data) + len(data) % 2  # IFDs start on a word boundary
        holder = Image.Exif()
        holder.endian = "<"
        for k, v in sub_ifds.items():
            holder[k] = dict(v)
        ifds = holder.tobytes(offset=start)[6 + 8:]  # skip "Exif\0\0" and the TIFF header
        n = struct.unpack_from("<H", ifds)[0]
        pointers = {}
        for i in range(n):
            tag, _type, _count, value = struct.unpack_from("<HHII", ifds, 2 + 12 * i)
            pointers[tag] = value
        again = tiff(pointers)
        if len(again) != len(data):
            raise RuntimeError("TIFF layout changed between passes")
        data = again + b"\0" * (start - len(again)) + ifds
    path.write_bytes(data)
