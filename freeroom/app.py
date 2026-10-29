"""Freeroom GUI: open a photo, develop it, enhance it with AI, save.

Raw tab: instant, per-pixel dials on the linear photo, then an optional HDR
section (local tone mapping).
B&W tab: more development, applied after the Raw tab.
Enhancement tab: AI denoise + sharpen, shown on top of the development.
Each AI model runs at most once per photo; its result is kept and the dials
are applied on top (see pipeline.py), so only a new photo re-runs the AI.
"""

from __future__ import annotations

import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon
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
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import develop, engine, imageio, pipeline, presets
from .widgets import Histogram, PhotoView, ResetSlider

SAVE_DIR = Path.home() / "Pictures" / "freeroom"
ICON = Path(__file__).resolve().parent / "assets" / "freeroom.png"  # optional, gitignored
MAX_RESULTS = 6  # AI results kept per photo (each is a full-size float image)

# Slider specs: key, label, min, max, default (slider units = value * 100), display format
DEV_LIGHT = [
    ("exposure", "Exposure", -300, 300, 0, lambda v: f"{v / 100:+.2f} EV"),
    ("shadows", "Shadows", -100, 100, 0, lambda v: f"{v:+d}"),
    ("midtones", "Midtones", -100, 100, 0, lambda v: f"{v:+d}"),
    ("highlights", "Highlights", -100, 100, 0, lambda v: f"{v:+d}"),
    ("contrast", "Contrast", -100, 100, 0, lambda v: f"{v:+d}"),
]
DEV_COLOUR = [
    ("temperature", "Temperature", -100, 100, 0, lambda v: f"{v:+d}"),
    ("tint", "Tint", -100, 100, 0, lambda v: f"{v:+d}"),
    ("saturation", "Saturation", -100, 100, 0, lambda v: f"{v:+d}"),
]
DEV_HDR = [
    ("hdr_compress", "Compression", 0, 100, 50, lambda v: f"{v}%"),
    ("hdr_detail", "Local detail", -100, 100, 30, lambda v: f"{v:+d}"),
]
BW_SWATCH = {"red": "#e04040", "orange": "#f08a24", "yellow": "#e8d030", "green": "#40b050",
             "aqua": "#30c0c0", "blue": "#4070e0", "purple": "#8a50d0", "magenta": "#d040b0"}
# per colour: its slider, then High / Low underneath
DEV_BW = [
    (f"bw_{name}{part}",
     f"<span style='color:{BW_SWATCH[name]}'>■</span> {name.capitalize()}" if not part
     else ("      High" if part == "_hi" else "      Low"),
     -100, 100, 0, lambda v: f"{v:+d}")
    for name, _ in develop.BW_COLOURS for part in develop.BW_PARTS
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
RAW_KEYS = [s[0] for s in DEV_LIGHT + DEV_COLOUR]
HDR_KEYS = [s[0] for s in DEV_HDR]
BW_KEYS = [s[0] for s in DEV_BW]
DEV_KEYS = RAW_KEYS + HDR_KEYS + BW_KEYS
ENH_KEYS = [s[0] for s in ENH_NR + ENH_SH]
SLIDER_DEFAULTS = {s[0]: s[4] for s in DEV_LIGHT + DEV_COLOUR + DEV_HDR + DEV_BW + ENH_NR + ENH_SH}

GROUP_STYLE = (
    "QGroupBox { font-weight: bold; font-size: 11pt; margin-top: 8px; padding-top: 10px; }"
    "QGroupBox * { font-weight: normal; font-size: 10pt; }"
)


class Worker(QObject):
    """Loads the models and runs them, off the GUI thread."""

    progress = Signal(int, int)
    stage = Signal(object)  # key of the stage that started
    stage_done = Signal(object)  # key of the stage whose result is complete
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

    def run(self, stages: list[tuple]):
        """Run (key, label, model, src, dst) stages in order. dst starts as a copy of src."""
        try:
            t0 = time.monotonic()
            for key, label, path, src, dst in stages:
                model = self._model(path)
                dst[:] = src
                self.stage.emit(key)
                self.status.emit(f"{label} with {path.stem}…")
                if not model.run(src, dst, progress=self.progress.emit, cancelled=lambda: self.cancel):
                    self.finished.emit(False, 0.0)
                    return
                if label == "Sharpening":
                    engine.keep_detail(src, dst)
                self.stage_done.emit(key)
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
        # AI results for the current photo, in the pipeline encoding, keyed by
        # (photo_gen, step, model…). Development settings are not part of the
        # key: the dials are applied on top, so results stay valid.
        self.g = 1.0  # pipeline encoding scale for this photo
        self.enc_orig: np.ndarray | None = None  # the photo in that encoding
        self.results: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.running: dict[tuple, np.ndarray] = {}  # key -> buffer being filled
        self.started: set[tuple] = set()
        self.busy = False
        self.pending = False
        self.cancelled = False  # the user stopped the running job: don't restart it by itself
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
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Stop the AI denoise / sharpen run (after the current tile)")
        self.cancel_btn.clicked.connect(self.cancel_job)
        self.cancel_btn.hide()
        self.status = QLabel(engine.device_name())

        top = QHBoxLayout()
        for w in (open_btn, self.save_btn, QLabel("   Preset:"), self.preset_box, save_preset, del_preset):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.status)
        top.addWidget(self.progress)
        top.addWidget(self.cancel_btn)

        # --- tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_develop_tab(), "Raw")
        self.tabs.addTab(self._build_bw_tab(), "B&&W")
        self.enh_tab = self._build_enhance_tab()
        self.tabs.addTab(self.enh_tab, "Enhancement")
        self.tabs.currentChanged.connect(self.tab_changed)

        hint = QLabel(
            "<b>Loupe</b> = 100% (1 photo pixel = 1 screen pixel).<br>"
            "Hold mouse button: before, in loupe · Hold Space: before, everywhere · "
            "Scroll: loupe size.<br>"
            "<i>Before</i> = as shot on the Raw and B&amp;W tabs, unenhanced on the Enhancement tab."
        )
        hint.setStyleSheet("color: #999;")
        hint.setWordWrap(True)

        side = QVBoxLayout()
        side.addWidget(QLabel("<b>Range</b> (R G B)"))
        side.addWidget(self.hist)
        side.addWidget(self.tabs, 1)
        side.addWidget(hint)
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
        self.worker.stage_done.connect(self.on_stage_done)
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
        s = ResetSlider(default)
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
        self.hdr_group = self._dev_group("HDR tone mapping", "hdr")
        self.hdr_group.setToolTip(
            "Squeezes a high-contrast scene into the picture: Compression lifts shadows and "
            "pulls highlights down area by area, Local detail brings back (or tames) texture "
            "and local contrast. Works best on RAW files."
        )
        form = QFormLayout(self.hdr_group)
        for spec in DEV_HDR:
            self._add_slider(form, spec, self.dev_changed)
        reset = QPushButton("Reset development")
        reset.clicked.connect(lambda: self._reset(RAW_KEYS + HDR_KEYS))
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.addWidget(light)
        lay.addWidget(colour)
        lay.addWidget(self.hdr_group)
        lay.addWidget(reset)
        lay.addStretch(1)
        return tab

    def _dev_group(self, title: str, flag: str) -> QGroupBox:
        """Checkable section that switches a development step (dev_params[flag]) on/off."""
        group = QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(bool(develop.DEFAULTS[flag]))

        def toggled(on):
            self.dev_params[flag] = on
            self.view.set_dev_params(self.dev_params)

        group.toggled.connect(toggled)
        return group

    def _build_bw_tab(self) -> QWidget:
        self.bw_group = self._dev_group("Black && white", "bw")
        form = QFormLayout(self.bw_group)
        for spec in DEV_BW:
            self._add_slider(form, spec, self.dev_changed)
        reset = QPushButton("Reset colour mix")
        reset.clicked.connect(lambda: self._reset(BW_KEYS))
        note = QLabel(
            "Each colour slider makes that colour lighter or darker in the black & white "
            "image; <i>High</i> / <i>Low</i> do the same only where it is bright / dark."
        )
        note.setStyleSheet("color: #999;")
        note.setWordWrap(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.addWidget(self.bw_group)
        lay.addWidget(reset)
        lay.addWidget(note)
        lay.addStretch(1)
        scroll = QScrollArea()  # 24 sliders don't fit the side panel
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        return scroll

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
            "Each AI model runs once per photo; development and these sliders "
            "are applied on top instantly. <i>Defocus</i> = soft / slightly out of focus, "
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
            dev = s.get("development", {})
            for key in DEV_KEYS:
                set_slider(key, dev.get(key, develop.DEFAULTS[key]))
            self.hdr_group.setChecked(bool(dev.get("hdr", develop.DEFAULTS["hdr"])))
            self.bw_group.setChecked(bool(dev.get("bw", develop.DEFAULTS["bw"])))
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
        self.g = pipeline.ai_scale(photo.linear, photo.baseline_ev)
        self.enc_orig = None
        self.results.clear()
        self.running.clear()
        self.started.clear()
        self.save_target = None
        self.save_btn.setEnabled(True)
        h, w, _ = photo.linear.shape
        kind = f"RAW, base exposure +{photo.baseline_ev:.2f} EV" if photo.meta.get("raw") else f"{photo.meta.get('bits')}-bit"
        self.setWindowTitle(f"Freeroom — {path.name} ({w}×{h}, {kind})")
        self.view.set_linear(photo.linear, photo.baseline_ev, self.g)
        if not self.busy:
            self.update_pipeline()

    def _keys(self) -> tuple[tuple | None, tuple | None]:
        """Result keys the current settings need: (denoise, sharpen)."""
        dn_model = self.denoise_box.currentData() if self.nr_group.isChecked() else None
        sh_model = self.sharpen_box.currentData() if self.sh_group.isChecked() else None
        dn_key = (self.photo_gen, "dn", dn_model) if dn_model is not None else None
        sh_key = (self.photo_gen, "sh", sh_model, dn_model) if sh_model is not None else None
        return dn_key, sh_key

    def _layer(self, key: tuple | None) -> np.ndarray | None:
        if key is None:
            return None
        if key in self.results:
            self.results.move_to_end(key)
            return self.results[key]
        if key in self.started:
            return self.running.get(key)  # partially filled, shown as it fills in
        return None

    def _show_layers(self):
        dn_key, sh_key = self._keys()
        self.view.set_layers(self._layer(dn_key), self._layer(sh_key))

    def update_pipeline(self):
        """Run the AI steps the current settings need that haven't run yet for
        this photo (only when the Enhancement tab is shown or a save waits),
        then save if a save is waiting."""
        if self.photo is None or self.denoise_box.count() == 0:
            return
        if self.tabs.currentWidget() is not self.enh_tab and self.save_target is None:
            return
        self._show_layers()
        if self.busy:
            # Let the running job finish (its result is kept), then come back here.
            self.pending = True
            return
        dn_key, sh_key = self._keys()
        stages = []
        if dn_key is not None and dn_key not in self.results or sh_key is not None and sh_key not in self.results:
            if self.enc_orig is None:
                self.enc_orig = pipeline.encode(self.photo.linear, self.g)
        if dn_key is not None and dn_key not in self.results:
            dst = self.enc_orig.copy()  # shows the unprocessed photo until tiles land
            self.running[dn_key] = dst
            stages.append((dn_key, "Denoising", dn_key[2], self.enc_orig, dst))
        if sh_key is not None and sh_key not in self.results:
            src = self.enc_orig if dn_key is None else self.results.get(dn_key, self.running.get(dn_key))
            dst = np.empty_like(src)  # filled by the worker once that stage starts
            self.running[sh_key] = dst
            stages.append((sh_key, "Sharpening", sh_key[2], src, dst))

        if not stages:
            if self.save_target is not None:
                self._do_save()
            return
        self.worker.cancel = False
        self.cancelled = False
        self.busy = True
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.show()
        self.refresh.start()
        self._show_layers()
        self.start_job.emit(stages)

    def on_stage(self, key: tuple):
        self.started.add(key)
        self._show_layers()

    def on_stage_done(self, key: tuple):
        dst = self.running.pop(key, None)
        self.started.discard(key)
        if dst is None or key[0] != self.photo_gen:
            return  # from a photo that has since been closed
        self.results[key] = dst
        while len(self.results) > MAX_RESULTS:
            self.results.popitem(last=False)
        self._show_layers()

    def on_progress(self, done: int, total: int):
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.progress.setFormat(f"tile {done}/{total}")

    def cancel_job(self):
        if not self.busy:
            return
        self.worker.cancel = True
        self.cancelled = True
        self.pending = False
        self.save_target = None  # a save waiting for this run is dropped too
        self.cancel_btn.setEnabled(False)
        self.status.setText("Cancelling after the current tile…")

    def _job_ended(self):
        self.busy = False
        self.refresh.stop()
        self.progress.hide()
        self.cancel_btn.hide()
        self.running.clear()
        self.started.clear()

    def on_finished(self, ok: bool, seconds: float):
        self._job_ended()
        if ok:
            self.status.setText(f"Done in {seconds:.1f} s · {engine.device_name()}")
        asked_again, self.pending = self.pending, False
        self._show_layers()
        if self.cancelled and not ok and not asked_again:
            self.status.setText("Cancelled. Change a model or reopen the Enhancement tab to run it again.")
            return
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

    def _do_save(self):
        out, self.save_target = self.save_target, None
        dn_key, sh_key = self._keys()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            rgb = pipeline.compose(
                self.photo.linear,
                self.results.get(dn_key),
                self.results.get(sh_key),
                self.g,
                self.dev_params,
                self.photo.baseline_ev,
                self.effective_enh(),
            )
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
    if ICON.exists():  # local only, not in the repo
        app.setWindowIcon(QIcon(str(ICON)))
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.open_path(Path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
