"""Batch processing: develop + enhance every photo in a folder with one preset.

Each photo goes through the same steps as an interactive save: load, the AI
steps the preset turns on (each once), development + slider blending, write.
Crops are per photo, so a batch never crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import develop, engine, imageio, pipeline

FORMATS = {"JPEG": ".jpg", "TIFF 16-bit": ".tif", "PNG 16-bit": ".png"}


@dataclass
class Job:
    dev: dict  # development params
    enh: dict  # effective enhancement sliders (sharpening off -> 0)
    denoise: Path | None  # AI model to run, None = step off
    sharpen: Path | None


def job_from_settings(settings: dict, enh_defaults: dict) -> Job:
    """Preset / settings dict (MainWindow.settings() format) -> Job. Missing keys
    get the defaults; a model that isn't installed raises ValueError."""
    dev = dict(develop.DEFAULTS)
    dev.update({k: v for k, v in settings.get("development", {}).items() if k in develop.DEFAULTS})
    enh = dict(enh_defaults)
    sections = settings.get("enhancement", {})
    nr = sections.get("noise_reduction", {})
    sh = sections.get("sharpening", {})
    for section in (nr, sh):
        enh.update({k: float(v) for k, v in section.items() if k in enh_defaults})

    def model(kind: str, section: dict, default_on: bool) -> Path | None:
        if not section.get("enabled", default_on):
            return None
        models = engine.list_models(kind)
        stem = section.get("model", models[0].stem if models else None)
        if stem is None:  # e.g. sharpening "None (classic only)"
            return None
        for p in models:
            if p.stem == stem:
                return p
        raise ValueError(f"{kind} model '{stem}' isn't installed (models/{kind}/)")

    sharpening_on = sh.get("enabled", False)
    if not sharpening_on:
        enh["ai_sharpen"] = enh["sharpen"] = 0.0
    return Job(dev, enh, model("denoise", nr, True), model("sharpen", sh, False))


def find_photos(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in imageio.ALL_EXTS)


def output_path(photo: Path, in_root: Path, out_root: Path, ext: str, suffix: str) -> Path:
    """Same relative place (subfolders kept), stem + suffix, new extension."""
    rel = photo.relative_to(in_root)
    return out_root / rel.parent / f"{rel.stem}{suffix}{ext}"


class Processor:
    """Runs jobs; keeps the AI models loaded between photos."""

    def __init__(self):
        self.models: dict[Path, engine.Restorer] = {}

    def _model(self, path: Path) -> engine.Restorer:
        if path not in self.models:
            self.models[path] = engine.Restorer(path)
        return self.models[path]

    def process(
        self,
        src: Path,
        out: Path,
        job: Job,
        stage: Callable[[str], None] = lambda s: None,
        tiles: Callable[[int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        """Develop src into out. Returns False if cancelled (nothing written)."""
        stage("Loading")
        photo = imageio.load(src)
        g = pipeline.ai_scale(photo.linear, photo.baseline_ev)
        dn = sharp = None
        if job.denoise or job.sharpen:
            enc = pipeline.encode(photo.linear, g)
            if job.denoise:
                stage("Denoising")
                dn = enc.copy()
                if not self._model(job.denoise).run(enc, dn, progress=tiles, cancelled=cancelled):
                    return False
            if job.sharpen:
                stage("Sharpening")
                base = dn if dn is not None else enc
                sharp = base.copy()
                if not self._model(job.sharpen).run(base, sharp, progress=tiles, cancelled=cancelled):
                    return False
                engine.keep_detail(base, sharp)
        if cancelled and cancelled():
            return False
        stage("Developing")
        rgb = pipeline.compose(photo.linear, dn, sharp, g, job.dev, photo.baseline_ev, job.enh)
        stage("Writing")
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.save(photo, np.clip(rgb, 0, 1), out)
        return True
