"""Looks: colour LUTs (.cube files and Hald CLUT images) applied after development.

A LUT maps display sRGB to display sRGB, which is what development outputs,
so it is the last step of `develop.develop`, blended in by its strength.

Formats:
- .cube (Adobe / Resolve): 3D (LUT_3D_SIZE) or 1D (LUT_1D_SIZE), with optional
  DOMAIN_MIN / DOMAIN_MAX. Red changes fastest in the table.
- Hald CLUT (.png / .tif): a square image of level L, L³ × L³ pixels, holding
  a cube of L² steps per colour (the format of RawTherapee's film simulations).

LUT files live in LUT_DIR; settings name them by file name there, or by full
path for a file elsewhere.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

LUT_DIR = Path.home() / ".config" / "freeroom" / "luts"
EXTS = {".cube", ".png", ".tif", ".tiff"}
_CHUNK = 1 << 22  # pixels per step when applying, to keep temporaries small


@dataclass
class Lut:
    table: np.ndarray  # 3D: (n, n, n, 3) indexed [b, g, r]; 1D: (n, 3)
    dmin: np.ndarray  # input domain per channel
    dmax: np.ndarray
    title: str = ""

    @property
    def is_3d(self) -> bool:
        return self.table.ndim == 4


def names() -> list[str]:
    """LUT files in LUT_DIR (subfolders included), as names relative to it."""
    if not LUT_DIR.is_dir():
        return []
    found = [p for p in LUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    return sorted((p.relative_to(LUT_DIR).as_posix() for p in found), key=str.lower)


def resolve(name: str) -> Path:
    p = Path(name).expanduser()
    return p if p.is_absolute() else LUT_DIR / p


def setting_name(path: Path) -> str:
    """How a settings dict refers to this file: by name inside LUT_DIR (so
    presets work on another computer with the same LUTs), else by full path."""
    path = Path(path).resolve()
    try:
        return path.relative_to(LUT_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def label(name: str) -> str:
    return Path(name).stem


# --- reading ----------------------------------------------------------------
def read_cube(path: Path) -> Lut:
    size3 = size1 = None
    dmin, dmax = np.zeros(3), np.ones(3)
    title = ""
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        head = line.split()[0].upper()
        if head[0].isdigit() or head[0] in "+-.":
            rows.append(line)
        elif head == "TITLE":
            title = line[5:].strip().strip('"')
        elif head == "LUT_3D_SIZE":
            size3 = int(line.split()[1])
        elif head == "LUT_1D_SIZE":
            size1 = int(line.split()[1])
        elif head == "DOMAIN_MIN":
            dmin = np.array(line.split()[1:4], float)
        elif head == "DOMAIN_MAX":
            dmax = np.array(line.split()[1:4], float)
        elif head in ("LUT_3D_INPUT_RANGE", "LUT_1D_INPUT_RANGE"):
            lo, hi = (float(v) for v in line.split()[1:3])
            dmin, dmax = np.full(3, lo), np.full(3, hi)
        # other keywords (e.g. LUT_IN_VIDEO_RANGE) don't change the table
    if size3 is None and size1 is None:
        raise ValueError("no LUT_3D_SIZE or LUT_1D_SIZE: not a .cube LUT")
    data = np.array([r.split()[:3] for r in rows], np.float32)
    n = size3 or size1
    want = n**3 if size3 else n
    if len(data) != want or n < 2:
        raise ValueError(f"expected {want} table rows, found {len(data)}")
    table = data.reshape(n, n, n, 3) if size3 else data
    if np.any(dmax <= dmin):
        raise ValueError("DOMAIN_MAX must be above DOMAIN_MIN")
    return Lut(table, dmin.astype(np.float32), dmax.astype(np.float32), title)


def read_hald(path: Path) -> Lut:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("can't read the image")
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("a Hald CLUT is a colour image")
    h, w = img.shape[:2]
    level = round(w ** (1 / 3))
    if h != w or level**3 != w or level < 2:
        raise ValueError(f"{w}×{h} isn't a Hald CLUT (L³ × L³ pixels, e.g. 512×512)")
    scale = np.float32(65535 if img.dtype == np.uint16 else 255)
    rgb = img[..., 2::-1].astype(np.float32) / scale  # BGR(A) -> RGB
    n = level * level
    return Lut(rgb.reshape(n, n, n, 3), np.zeros(3, np.float32), np.ones(3, np.float32), Path(path).stem)


def read(path: Path) -> Lut:
    path = Path(path)
    if path.suffix.lower() == ".cube":
        return read_cube(path)
    return read_hald(path)


_cache: dict[tuple, Lut] = {}
_lock = threading.Lock()


def get(name: str) -> Lut:
    """The LUT a setting names, read once (again when the file changes).
    Raises FileNotFoundError / ValueError."""
    path = resolve(name)
    key = (str(path), path.stat().st_mtime_ns)
    with _lock:
        lut = _cache.get(key)
    if lut is None:
        lut = read(path)
        with _lock:
            for k in [k for k in _cache if k[0] == key[0]]:
                del _cache[k]
            _cache[key] = lut
    return lut


def problem(name: str) -> str | None:
    """Why the LUT a setting names can't be used, or None if it can."""
    if not name:
        return None
    try:
        get(name)
        return None
    except FileNotFoundError:
        return f"LUT “{label(name)}” not found ({resolve(name)})"
    except Exception as e:
        return f"LUT “{label(name)}”: {e}"


# --- applying -----------------------------------------------------------------
_atlases: dict[int, np.ndarray] = {}


def _atlas(t: np.ndarray) -> np.ndarray:
    """The 3D table as one 2D image: the red-green slice of each blue step side
    by side, so cv2.remap can do the red-green part of the lookup."""
    key = id(t)
    a = _atlases.get(key)
    if a is None or a.shape[0] != t.shape[0]:
        n = t.shape[0]
        a = np.ascontiguousarray(t.transpose(1, 0, 2, 3).reshape(n, n * n, 3))
        _atlases.clear()  # one look at a time in practice
        _atlases[key] = a
    return a


def _apply_3d(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Trilinear lookup of x (h, w, 3) in 0..1 in table t (n, n, n, 3): bilinear in
    red-green by cv2.remap (1/32-cell steps, finer than 8-bit output) on the two
    nearest blue slices, then linear between them. OpenCV does the arithmetic
    too, as it uses every core."""
    n = t.shape[0]
    atlas = _atlas(t)
    ur, ug, ub = cv2.split(cv2.multiply(x, (n - 1.0,) * 3))
    b0 = np.minimum(ub, np.float32(n - 2 + 0.5)).astype(np.int32).astype(np.float32)  # floor (ub >= 0)
    fb = cv2.subtract(ub, b0)
    mx = cv2.scaleAdd(b0, float(n), ur)
    lo = cv2.remap(atlas, mx, ug, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    hi = cv2.remap(atlas, cv2.add(mx, float(n)), ug, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.add(lo, cv2.multiply(cv2.subtract(hi, lo), cv2.merge((fb, fb, fb))))


def _apply_1d(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, len(t), dtype=np.float32)
    return np.stack([np.interp(x[..., c], grid, t[:, c]) for c in range(3)], axis=-1).astype(np.float32)


def apply(rgb: np.ndarray, lut: Lut, strength: float = 1.0) -> np.ndarray:
    """Display sRGB (h, w, 3) through the LUT, blended in by strength (0..1)."""
    shape = rgb.shape
    x = np.asarray(rgb, np.float32)
    if x.ndim != 3 or x.shape[1] >= 32767:  # cv2.remap wants 2D maps of moderate width
        x = x.reshape(-1, 1, 3)
    plain = not (np.any(lut.dmin != 0) or np.any(lut.dmax != 1))
    out = np.empty(x.shape, np.float32)
    rows = max(1, _CHUNK // x.shape[1])
    s = np.float32(strength)
    for a in range(0, x.shape[0], rows):
        part = x[a:a + rows]
        u = part if plain else (part - lut.dmin) / (lut.dmax - lut.dmin)
        u = np.clip(u, 0.0, 1.0)
        y = _apply_3d(u, lut.table) if lut.is_3d else _apply_1d(u, lut.table)
        out[a:a + rows] = cv2.addWeighted(y, float(s), part, float(1 - s), 0.0) if s < 1 else y
    np.clip(out, 0.0, 1.0, out=out)
    return out.reshape(shape)
