"""Edits kept with the photo: a small JSON file next to it.

`IMG_1234.CR2` gets `IMG_1234.CR2.freeroom.json` beside it, holding everything
needed to show the photo again as it was left: the Base settings (development,
look/LUT, enhancement), the crop, the layers with their selections, and the
red-eye fixes. The photo itself is never changed. Opening a photo that has
one restores it.

Arrays (selections, red-eye masks) are stored as zlib-compressed .npy data in
base64, so the file stays one self-contained, readable JSON document.
"""

from __future__ import annotations

import base64
import io
import json
import os
import zlib
from pathlib import Path

import numpy as np

SUFFIX = ".freeroom.json"
VERSION = 1


def path_for(photo: Path) -> Path:
    photo = Path(photo)
    return photo.with_name(photo.name + SUFFIX)


def pack(a: np.ndarray | None) -> str | None:
    if a is None:
        return None
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
    return base64.b64encode(zlib.compress(buf.getvalue(), 6)).decode("ascii")


def unpack(s: str | None) -> np.ndarray | None:
    if not s:
        return None
    return np.load(io.BytesIO(zlib.decompress(base64.b64decode(s))), allow_pickle=False)


def dumps(edit: dict) -> str:
    return json.dumps({"version": VERSION, **edit}, indent=1) + "\n"


def read(photo: Path) -> dict | None:
    """The kept edit of a photo, or None if it has none. Raises on a broken file."""
    p = path_for(photo)
    if not p.is_file():
        return None
    data = json.loads(p.read_text())
    if not isinstance(data, dict) or "settings" not in data:
        raise ValueError(f"{p.name} isn't a Freeroom edit")
    return data


def same(photo: Path, edit: dict) -> bool:
    """The photo's kept edit is exactly this one."""
    p = path_for(photo)
    try:
        return p.read_text() == dumps(edit)
    except OSError:
        return False


def write(photo: Path, edit: dict) -> Path:
    """Write (atomically) the kept edit for the photo."""
    p = path_for(photo)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(dumps(edit))
    os.replace(tmp, p)
    return p


def remove(photo: Path):
    path_for(photo).unlink(missing_ok=True)
