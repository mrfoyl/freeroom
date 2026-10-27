"""Freeroom GUI: open a photo, develop it, enhance it with AI, save.

Pipeline: linear RAW/JPEG -> Raw development tab (instant, per-pixel)
-> Enhancement tab (AI denoise + sharpen on the developed photo) -> save.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import develop, engine, imageio, presets
from .widgets import Histogram, PhotoView

SAVE_DIR = Path.home() / "Pictures" / "freeroom"

# Slider specs: key, label, min, max, default (slider units = value * 100), display format
DEV_LIGHT = [
    ("exposure", "Exposure", -300, 300, 0, lambda v: f"{v / 100:+.2f} EV"),
    ("shadows", "Shadows", -100, 100, 0, lambda v: f"{v:+d}"),
    ("midtones", "Midtones", -100, 100, 0, lambda v: f"{v:+d}"),
    ("highlights", "Highlights", -100, 100, 0, lambda v: f"{v:+d}"),
]
DEV_COLOUR = [
    ("temperature", "Temperature", -100, 100, 0, lambda v: f"{v:+d}"),
    ("tint", "Tint", -100, 100, 0, lambda v: f"{v:+d}"),
    ("saturation", "Saturation", -100, 100, 0, lambda v: f"{v:+d}"),
]
PCT = lambda v: f"{v}%"  # noqa: E731
ENH_NR = [
    ("luminance", "Luminance noise", 0, 100, 85, PCT),
    ("color", "Colour noise", 0, 100, 100, PCT),
]
ENH_SH = [
    ("ai_sharpen", "AI sharpen", 0, 150, 70, PCT),  # >100% exaggerates the deblur model
    ("sharpen", "Classic sharpen", 0, 100, 15, PCT),
]
DEV_KEYS = [s[0] for s in DEV_LIGHT + DEV_COLOUR]
ENH_KEYS = [s[0] for s in ENH_NR + ENH_SH]
SLIDER_DEFAULTS = {s[0]: s[4] for s in DEV_LIGHT + DEV_COLOUR + ENH_NR + ENH_SH}

GROUP_STYLE = (
    "QGroupBox { font-weight: bold; font-size: 11pt; margin-top: 8px; padding-top: 10px; }"
    "QGroupBox * { font-weight: normal; font-size: 10pt; }"
)


class Worker(QObject):
    """Loads the models and runs them, off the GUI thread."""

    progress = Signal(int, int)
    stage = Signal(str)
    status = Signal(str)
    finished = Signal(bool, float)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.models: dict[Path, engine.Restorer] = {}  # kept loaded; they're small
        self.cancel = False

    def _model(self, path: Path) -> engine.Restorer:
        if path not in self.models:
            self.status.emit(f"Loading model {path.stem}…")
            self.models[path] = engine.Restorer(path)
        return self.models[path]

    def run(self, stages: list[tuple[str, Path, np.ndarray, np.ndarray]]):
        """Run (label, model, src, dst) stages in order. dst starts as a copy of src."""
        try:
            t0 = time.monotonic()
            for label, path, src, dst in stages:
                model = self._model(path)
                dst[:] = src
                self.stage.emit(label)
                self.status.emit(f"{label} with {path.stem}…")
                if not model.run(src, dst, progress=self.progress.emit, cancelled=lambda: self.cancel):
                    self.finished.emit(False, 0.0)
                    return
                if label == "Sharpening":
                    engine.keep_detail(src, dst)
            self.finished.emit(True, time.monotonic() - t0)
        except Exception as e:  # surface anything (OOM, bad model file) in the UI
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    start_job = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Freeroom")
        self.resize(1500, 950)
        self.setAcceptDrops(True)
        self.photo: imageio.Photo | None = None
        self.photo_gen = 0  # bumped per opened photo; part of every cache key
        self.dev_params = dict(develop.DEFAULTS)
        self.enh_params = {k: SLIDER_DEFAULTS[k] / 100 for k in ENH_KEYS}
        # Computed images, each tagged with the settings that produced it so
        # nothing re-runs unless those settings changed.
        self.developed: np.ndarray | None = None
        self.developed_key: tuple | None = None
        self.ai: np.ndarray | None = None  # denoised developed photo
        self.ai_key: tuple | None = None
        self.sharp: np.ndarray | None = None  # deblurred (denoised or developed) photo
        self.sharp_key: tuple | None = None
        self.job_keys: tuple | None = None  # (ai_key, sharp_key) of the running job
        self.busy = False
        self.pending = False
        self.save_target: Path | None = None
        self._applying = False  # loading a preset: hold back pipeline runs

        self.view = PhotoView()
        self.hist = Histogram()
        self.view.rendered.connect(self.on_rendered)
        self.sliders: dict[str, tuple[QSlider, QLabel]] = {}

        # --- top bar
        open_btn = QPushButton("Open…")
        open_btn.setShortcut("Ctrl+O")
        open_btn.clicked.connect(self.open_dialog)
        self.save_btn = QPushButton("Save…")
        self.save_btn.setShortcut("Ctrl+S")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_dialog)

        self.preset_box = QComboBox()
        self.preset_box.setMinimumWidth(180)
        self.preset_box.activated.connect(self.preset_chosen)
        save_preset = QPushButton("Save preset…")
        save_preset.clicked.connect(self.save_preset)
        del_preset = QPushButton("Delete preset")
        del_preset.clicked.connect(self.delete_preset)
        self.refresh_presets()

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setTextVisible(True)
        self.progress.hide()
        self.status = QLabel(engine.device_name())

        top = QHBoxLayout()
        for w in (open_btn, self.save_btn, QLabel("   Preset:"), self.preset_box, save_preset, del_preset):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.status)
        top.addWidget(self.progress)

        # --- tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_develop_tab(), "Raw development")
        self.enh_tab = self._build_enhance_tab()
        self.tabs.addTab(self.enh_tab, "Enhancement")
        self.tabs.currentChanged.connect(self.tab_changed)

        hint = QLabel(
            "<b>Loupe</b> = 100% (1 photo pixel = 1 screen pixel).<br>"
            "Hold mouse button: before, in loupe · Hold Space: before, everywhere · "
            "Scroll: loupe size.<br>"
            "<i>Before</i> = as shot on the development tab, unenhanced on the enhancement tab."
        )
        hint.setStyleSheet("color: #999;")
        hint.setWordWrap(True)

        side = QVBoxLayout()
        side.addWidget(QLabel("<b>Range</b> (R G B)"))
        side.addWidget(self.hist)
        side.addWidget(self.tabs)
        side.addWidget(hint)
        side.addStretch(1)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(360)
        side_w.setStyleSheet(GROUP_STYLE)

        body = QHBoxLayout()
        body.addWidget(self.view, 1)
        body.addWidget(side_w)
        root = QVBoxLayout()
        root.addLayout(top)
        root.addLayout(body, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        # --- worker thread
        self.thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.start_job.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.stage.connect(self.on_stage)
        self.worker.status.connect(self.status.setText)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.thread.start()

        self.refresh = QTimer(self)
        self.refresh.setInterval(400)
        self.refresh.timeout.connect(self.view.ai_updated)

        self._sync_enh_view()
        if self.denoise_box.count() == 0:
            self.status.setText("No models in models/ — run setup/download-models.sh")

    # --- building the tabs ----------------------------------------------
    def _add_slider(self, form: QFormLayout, spec, on_change: Callable[[str, float], None]):
        key, text, lo, hi, default, fmt = spec
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(default)
        val = QLabel(fmt(default))
        val.setMinimumWidth(62)

        def changed(v):
            val.setText(fmt(v))
            on_change(key, v / 100)

        s.valueChanged.connect(changed)
        row = QHBoxLayout()
        row.addWidget(s)
        row.addWidget(val)
        form.addRow(text, row)
        self.sliders[key] = (s, val)

    def _build_develop_tab(self) -> QWidget:
        light = QGroupBox("Light")
        form = QFormLayout(light)
        for spec in DEV_LIGHT:
            self._add_slider(form, spec, self.dev_changed)
        colour = QGroupBox("Colour")
        form = QFormLayout(colour)
        for spec in DEV_COLOUR:
            self._add_slider(form, spec, self.dev_changed)
        reset = QPushButton("Reset development")
        reset.clicked.connect(lambda: self._reset(DEV_KEYS))
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.addWidget(light)
        lay.addWidget(colour)
        lay.addWidget(reset)
        lay.addStretch(1)
        return tab

    def _build_enhance_tab(self) -> QWidget:
        self.denoise_box = QComboBox()
        for p in engine.list_models("denoise"):
            self.denoise_box.addItem(engine.pretty_name(p), p)
        self.denoise_box.currentIndexChanged.connect(lambda _: self.enh_settings_changed())
        self.nr_group = QGroupBox("Noise reduction")
        self.nr_group.setCheckable(True)
        self.nr_group.setChecked(True)
        self.nr_group.toggled.connect(lambda _: self.enh_settings_changed())
        form = QFormLayout(self.nr_group)
        form.addRow("AI model", self.denoise_box)
        for spec in ENH_NR:
            self._add_slider(form, spec, self.enh_changed)

        self.sharpen_box = QComboBox()
        for p in engine.list_models("sharpen"):
            self.sharpen_box.addItem(engine.pretty_name(p), p)
        self.sharpen_box.addItem("None (classic only)", None)
        self.sharpen_box.currentIndexChanged.connect(lambda _: self.enh_settings_changed())
        self.sh_group = QGroupBox("Sharpening")
        self.sh_group.setCheckable(True)
        self.sh_group.setChecked(False)
        self.sh_group.toggled.connect(lambda _: self.enh_settings_changed())
        form = QFormLayout(self.sh_group)
        form.addRow("AI model", self.sharpen_box)
        for spec in ENH_SH:
            self._add_slider(form, spec, self.enh_changed)

        reset = QPushButton("Reset enhancement sliders")
        reset.clicked.connect(lambda: self._reset(ENH_KEYS))
        note = QLabel(
            "Applied to the developed photo. <i>Defocus</i> = soft / slightly out of focus, "
            "<i>Motion</i> = camera shake."
        )
        note.setStyleSheet("color: #999;")
        note.setWordWrap(True)
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.addWidget(self.nr_group)
        lay.addWidget(self.sh_group)
        lay.addWidget(reset)
        lay.addWidget(note)
        lay.addStretch(1)
        return tab

    # --- settings changes -----------------------------------------------
    def _reset(self, keys: list[str]):
        for key in keys:
            self.sliders[key][0].setValue(SLIDER_DEFAULTS[key])

    def dev_changed(self, key: str, value: float):
        self.dev_params[key] = value
        self.view.set_dev_params(self.dev_params)

    def enh_changed(self, key: str, value: float):
        self.enh_params[key] = value
        self.view.set_enh_params(self.effective_enh())

    def enh_settings_changed(self):
        self._sync_enh_view()
        if not self._applying:
            self.update_pipeline()

    def _sync_enh_view(self):
        self.sliders["ai_sharpen"][0].setEnabled(self.sharpen_box.currentData() is not None)
        self.view.set_enh_params(self.effective_enh())

    def effective_enh(self) -> dict:
        p = dict(self.enh_params)
        if not self.sh_group.isChecked():
            p["ai_sharpen"] = p["sharpen"] = 0.0
        return p

    def tab_changed(self, _):
        enhance = self.tabs.currentWidget() is self.enh_tab
        self.view.set_mode("enhance" if enhance else "develop")
        if enhance:
            self.update_pipeline()

    def on_rendered(self, rgb: np.ndarray):
        note = ""
        if self.photo is not None and self.photo.meta.get("raw") and self.view.mode == "develop":
            small = self.view.proxy_linear()
            frac = develop.beyond_white(small, self.dev_params, self.photo.baseline_ev) if small is not None else 0
            if frac >= 0.002:
                if self.dev_params["highlights"] < 0:
                    note = f"Highlights is recovering RAW detail past white ({frac * 100:.1f}%)"
                else:
                    note = f"RAW has detail past white ({frac * 100:.1f}%): pull Highlights left"
        self.hist.set_image(rgb, note)

    # --- presets ----------------------------------------------------------
    def settings(self) -> dict:
        def stem(box: QComboBox):
            p = box.currentData()
            return p.stem if p is not None else None

        e = self.enh_params
        return {
            "development": dict(self.dev_params),
            "enhancement": {
                "noise_reduction": {
                    "enabled": self.nr_group.isChecked(),
                    "model": stem(self.denoise_box),
                    "luminance": e["luminance"],
                    "color": e["color"],
                },
                "sharpening": {
                    "enabled": self.sh_group.isChecked(),
                    "model": stem(self.sharpen_box),
                    "ai_sharpen": e["ai_sharpen"],
                    "sharpen": e["sharpen"],
                },
            },
        }

    def apply_settings(self, s: dict):
        def set_slider(key, value):
            if value is not None:
                self.sliders[key][0].setValue(round(float(value) * 100))

        def set_model(box: QComboBox, stem):
            for i in range(box.count()):
                p = box.itemData(i)
                if (p.stem if p is not None else None) == stem:
                    box.setCurrentIndex(i)
                    return

        self._applying = True
        try:
            for key in DEV_KEYS:
                set_slider(key, s.get("development", {}).get(key, develop.DEFAULTS[key]))
            enh = s.get("enhancement", {})
            for group, box, section in (
                (self.nr_group, self.denoise_box, enh.get("noise_reduction", {})),
                (self.sh_group, self.sharpen_box, enh.get("sharpening", {})),
            ):
                if "enabled" in section:
                    group.setChecked(bool(section["enabled"]))
                if "model" in section:
                    set_model(box, section["model"])
                for key in ENH_KEYS:
                    if key in section:
                        set_slider(key, section[key])
        finally:
            self._applying = False
        self._sync_enh_view()
        self.update_pipeline()

    def refresh_presets(self, select: str | None = None):
        self.preset_box.clear()
        self.preset_box.addItem("— choose —", None)
        for name in presets.names():
            self.preset_box.addItem(name, name)
        if select:
            self.preset_box.setCurrentIndex(max(0, self.preset_box.findData(select)))

    def preset_chosen(self, index: int):
        name = self.preset_box.itemData(index)
        if not name:
            return
        try:
            self.apply_settings(presets.load(name))
            self.status.setText(f"Preset “{name}” loaded")
        except Exception as e:
            QMessageBox.warning(self, "Could not load preset", f"{name}: {e}")

    def save_preset(self):
        current = self.preset_box.currentData() or ""
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:", text=current)
        name = name.strip()
        if not ok or not name:
            return
        if name in presets.names() and name != current:
            if QMessageBox.question(self, "Save preset", f"Overwrite preset “{name}”?") != QMessageBox.StandardButton.Yes:
                return
        presets.save(name, self.settings())
        self.refresh_presets(select=name)
        self.status.setText(f"Preset “{name}” saved")

    def delete_preset(self):
        name = self.preset_box.currentData()
        if not name:
            return
        if QMessageBox.question(self, "Delete preset", f"Delete preset “{name}”?") != QMessageBox.StandardButton.Yes:
            return
        presets.delete(name)
        self.refresh_presets()

    # --- open / process -------------------------------------------------
    def open_dialog(self):
        exts = " ".join(f"*{e} *{e.upper()}" for e in sorted(imageio.ALL_EXTS))
        start = str(self.photo.path.parent) if self.photo else str(Path.home() / "Pictures")
        path, _ = QFileDialog.getOpenFileName(self, "Open photo", start, f"Photos ({exts})")
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            photo = imageio.load(path)
        except Exception as e:
            QMessageBox.warning(self, "Could not open", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        if self.busy:
            self.worker.cancel = True
            self.pending = True
        self.photo = photo
        self.photo_gen += 1
        self.developed = self.ai = self.sharp = None
        self.developed_key = self.ai_key = self.sharp_key = None
        self.save_target = None
        self.save_btn.setEnabled(True)
        h, w, _ = photo.linear.shape
        kind = f"RAW, base exposure +{photo.baseline_ev:.2f} EV" if photo.meta.get("raw") else f"{photo.meta.get('bits')}-bit"
        self.setWindowTitle(f"Freeroom — {path.name} ({w}×{h}, {kind})")
        self.view.set_linear(photo.linear, photo.baseline_ev)
        if not self.busy:
            self.update_pipeline()

    def _dev_key(self) -> tuple:
        return (self.photo_gen, *(self.dev_params[k] for k in DEV_KEYS))

    def _ensure_developed(self) -> np.ndarray:
        key = self._dev_key()
        if self.developed_key != key:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                # A new array, never in place: a running job may still read the old one.
                self.developed = develop.develop(self.photo.linear, self.dev_params, self.photo.baseline_ev)
            finally:
                QApplication.restoreOverrideCursor()
            self.developed_key = key
        return self.developed

    def _wanted(self) -> tuple[tuple | None, tuple | None]:
        """What the current settings need: (denoise key, sharpen key)."""
        dk = self._dev_key()
        ai_key = (self.denoise_box.currentData(), dk) if self.nr_group.isChecked() else None
        sharp_model = self.sharpen_box.currentData() if self.sh_group.isChecked() else None
        sharp_key = (sharp_model, ai_key, dk) if sharp_model is not None else None
        return ai_key, sharp_key

    def update_pipeline(self):
        """Bring the enhancement up to date with the settings (only when it's
        being looked at or saved), then save if a save is waiting."""
        if self.photo is None or self.denoise_box.count() == 0:
            return
        if self.tabs.currentWidget() is not self.enh_tab and self.save_target is None:
            return
        if self.busy:
            # Let the running job stop at the next tile, then come back here.
            if self.job_keys != self._wanted():
                self.worker.cancel = True
                self.pending = True
            return
        developed = self._ensure_developed()
        ai_key, sharp_key = self._wanted()
        stages = []
        if ai_key is not None and self.ai_key != ai_key:
            # The loupe shows the developed photo until each tile is done.
            self.ai = developed.copy()
            self.ai_key = None
            stages.append(("Denoising", ai_key[0], developed, self.ai))
        base = self.ai if ai_key is not None else developed
        if sharp_key is not None and self.sharp_key != sharp_key:
            self.sharp = np.empty_like(base)  # filled by the worker once that stage starts
            self.sharp_key = None
            stages.append(("Sharpening", sharp_key[0], base, self.sharp))
        shown_sharp = self.sharp if sharp_key is not None and self.sharp_key == sharp_key else None
        self.view.set_enhance(developed, base, shown_sharp)

        if not stages:
            if self.save_target is not None:
                self._do_save(developed, base, shown_sharp)
            return
        self.job_keys = (ai_key, sharp_key)
        self.worker.cancel = False
        self.busy = True
        self.progress.setValue(0)
        self.progress.show()
        self.refresh.start()
        self.start_job.emit(stages)

    def on_stage(self, label: str):
        if label == "Sharpening":
            ai_key, _ = self.job_keys
            if ai_key is not None:
                self.ai_key = ai_key  # denoise stage finished
            self.view.set_sharp(self.sharp)

    def on_progress(self, done: int, total: int):
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.progress.setFormat(f"tile {done}/{total}")

    def _job_ended(self):
        self.busy = False
        self.refresh.stop()
        self.progress.hide()
        self.view.ai_updated()

    def on_finished(self, ok: bool, seconds: float):
        self._job_ended()
        if ok:
            ai_key, sharp_key = self.job_keys
            if ai_key is not None:
                self.ai_key = ai_key
            if sharp_key is not None:
                self.sharp_key = sharp_key
            self.status.setText(f"Done in {seconds:.1f} s · {engine.device_name()}")
        self.pending = False
        self.update_pipeline()

    def on_failed(self, msg: str):
        self._job_ended()
        self.pending = False
        self.save_target = None
        self.status.setText("Failed")
        QMessageBox.critical(self, "Processing failed", msg)

    # --- save -----------------------------------------------------------
    def save_dialog(self):
        if self.photo is None:
            return
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        suggested = SAVE_DIR / f"{self.photo.path.stem}_edit.jpg"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save photo",
            str(suggested),
            "JPEG (*.jpg);;TIFF 16-bit (*.tif);;PNG 16-bit (*.png)",
        )
        if not path:
            return
        out = Path(path)
        if out.suffix.lower() not in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}:
            out = out.with_suffix(".jpg")
        if out.resolve() == self.photo.path.resolve():
            QMessageBox.warning(self, "Could not save", "Refusing to overwrite the original photo.")
            return
        # Runs whatever enhancement is still missing, then saves (see update_pipeline).
        self.save_target = out
        self.status.setText("Preparing full-size image for saving…")
        self.update_pipeline()

    def _do_save(self, developed: np.ndarray, base: np.ndarray, sharp: np.ndarray | None):
        out, self.save_target = self.save_target, None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            rgb = engine.adjust(developed, base, sharp, **self.effective_enh())
            saved = imageio.save(self.photo, rgb, out)
        except Exception as e:
            QMessageBox.warning(self, "Could not save", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.status.setText(f"Saved {saved}")

    # --- drag & drop ----------------------------------------------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.open_path(Path(urls[0]))

    def closeEvent(self, e):
        self.worker.cancel = True
        self.thread.quit()
        self.thread.wait(5000)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Freeroom")
    app.setDesktopFileName("freeroom")
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.open_path(Path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
