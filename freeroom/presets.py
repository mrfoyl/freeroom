"""Presets: every development + enhancement setting, as small JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path

PRESET_DIR = Path.home() / ".config" / "freeroom" / "presets"
VERSION = 1


def names() -> list[str]:
    if not PRESET_DIR.is_dir():
        return []
    return sorted((p.stem for p in PRESET_DIR.glob("*.json")), key=str.lower)


def _path(name: str) -> Path:
    safe = re.sub(r'[/\\:*?"<>|]', "_", name).strip() or "preset"
    return PRESET_DIR / f"{safe}.json"


def save(name: str, settings: dict) -> Path:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(name)
    path.write_text(json.dumps({"version": VERSION, **settings}, indent=2) + "\n")
    return path


def load(name: str) -> dict:
    return json.loads(_path(name).read_text())


def delete(name: str):
    _path(name).unlink(missing_ok=True)
