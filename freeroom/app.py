"""Freeroom GUI: open a photo, develop it, enhance it with AI, save.

Raw tab: instant, per-pixel dials on the linear photo, then an optional HDR
section (local tone mapping).
B&W tab: more development, applied after the Raw tab.
Enhancement tab: AI denoise + sharpen, shown on top of the development.
Crop (top bar): applied to the view and the saved file; kept per photo, not in presets.
Batch… (top bar): one preset over a whole folder (see batch.py).
Library (G) / Edit (E): browse folders with thumbnails and a preview (browser.py),
open a photo in the editor.
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
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import batch, develop, engine, imageio, pipeline, presets
from .browser import LibraryView
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
# Crop aspect ratios (width / height); "original" = the photo's own
CROP_ASPECTS = [("Free", None), ("Original", "original"), ("1:1", 1.0), ("3:2", 3 / 2),
                ("4:3", 4 / 3), ("5:4", 5 / 4), ("16:9", 16 / 9)]
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


class Loader(QObject):
    """Opens photos off the GUI thread (RAW decoding takes seconds)."""

    loaded = Signal(int, object, float)  # request id, Photo, AI encoding scale
    failed = Signal(int, str, str)  # request id, file name, error

    def load(self, req: int, path: Path):
        try:
            photo = imageio.load(path)
            self.loaded.emit(req, photo, pipeline.ai_scale(photo.linear, photo.baseline_ev))
        except Exception as e:
            self.failed.emit(req, path.name, str(e))


class Saver(QObject):
    """Develops the full-size photo and writes it, off the GUI thread."""

    progress = Signal(int, int, str)  # step done, steps, label of the step starting
    saved = Signal(str)
    failed = Signal(str)

    def save(self, job: dict):
        try:
            total = pipeline.compose_steps(job["dn"], job["sharp"]) + 1
            done = [0]

            def step(label):
                self.progress.emit(done[0], total, label)
                done[0] += 1

            lin, dn, sharp, base = job["photo"].linear, job["dn"], job["sharp"], None
            if job["crop"]:
                # develop only the cropped part; HDR still uses the whole photo's map
                x0, y0, x1, y1 = job["crop"]
                if job["dev"].get("hdr"):
                    base = develop.hdr_base(develop.hdr_map(lin), lin.shape, (y1 - y0, x1 - x0), (y0, x0, y1, x1))
                cut = np.s_[y0:y1, x0:x1]
                lin = lin[cut]
                dn = None if dn is None else dn[cut]
                sharp = None if sharp is None else sharp[cut]
            rgb = pipeline.compose(
                lin, dn, sharp, job["g"], job["dev"], job["photo"].baseline_ev, job["enh"],
                base=base, progress=step,
            )
            step(f"Writing {job['out'].name}")
            saved = imageio.save(job["photo"], rgb, job["out"])
            self.saved.emit(str(saved))
        except Exception as e:
            self.failed.emit(str(e))


class BatchWorker(QObject):
    """Runs a batch off the GUI thread, one photo after another."""

    file_started = Signal(int, int, str)  # index, count, name
    stage = Signal(str)  # "name: stage"
    tiles = Signal(int, int)
    file_done = Signal(str, str)  # status (done / skipped / failed), message
    finished = Signal(bool)  # False = cancelled

    def __init__(self):
        super().__init__()
        self.processor = batch.Processor()
        self.cancel = False

    def run(self, plan: dict):
        photos, job = plan["photos"], plan["job"]
        for i, src in enumerate(photos):
            if self.cancel:
                break
            out = batch.output_path(src, plan["in_dir"], plan["out_dir"], plan["ext"], plan["suffix"])
            name = str(src.relative_to(plan["in_dir"]))
            self.file_started.emit(i, len(photos), name)
            if out.exists() and not plan["overwrite"]:
                self.file_done.emit("skipped", f"{name}: {out.name} already exists")
                continue
            try:
                ok = self.processor.process(src, out, job, lambda st, n=name: self.stage.emit(f"{n}: {st}…"),
                                            self.tiles.emit, lambda: self.cancel)
                if ok:
                    self.file_done.emit("done", f"{name} → {out.relative_to(plan['out_dir'])}")
            except Exception as e:  # keep going with the next photo
                self.file_done.emit("failed", f"{name}: {type(e).__name__}: {e}")
        self.finished.emit(not self.cancel)


class BatchDialog(QDialog):
    """Pick a preset, an input and an output folder; process every photo."""

    start = Signal(object)

    def __init__(self, win: "MainWindow"):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Batch processing")
        self.setMinimumWidth(640)
        self.running = False

        self.preset = QComboBox()
        self.preset.addItem("Current settings", None)
        for name in presets.names():
            self.preset.addItem(name, name)
        current = win.preset_box.currentData()
        if current:
            self.preset.setCurrentIndex(max(0, self.preset.findData(current)))
        start_dir = str(win.photo.path.parent) if win.photo else str(Path.home() / "Pictures")
        self.in_dir = QLineEdit(start_dir)
        self.out_dir = QLineEdit(str(SAVE_DIR / "batch"))
        self.format = QComboBox()
        self.format.addItems(list(batch.FORMATS))
        self.suffix = QLineEdit("")
        self.suffix.setPlaceholderText("e.g. _edit (optional)")
        self.recursive = QCheckBox("Include subfolders (kept in the output)")
        self.overwrite = QCheckBox("Overwrite files that already exist (else skip them)")

        self.browse: list[QPushButton] = []

        def folder_row(edit: QLineEdit, title: str):
            pick = QPushButton("Browse…")
            pick.clicked.connect(lambda: self._pick(edit, title))
            self.browse.append(pick)
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(pick)
            return row

        form = QFormLayout()
        form.addRow("Preset", self.preset)
        form.addRow("Input folder", folder_row(self.in_dir, "Input folder"))
        form.addRow("Output folder", folder_row(self.out_dir, "Output folder"))
        form.addRow("Format", self.format)
        form.addRow("Name suffix", self.suffix)
        form.addRow("", self.recursive)
        form.addRow("", self.overwrite)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #999;")
        self.current = QLabel("")
        self.overall = QProgressBar()
        self.tile_bar = QProgressBar()
        self.tile_bar.setFormat("tile %v/%m")
        self.log = QListWidget()
        self.log.setMinimumHeight(160)
        self.start_btn = QPushButton("Start")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self.start_batch)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_batch)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        for b in (self.start_btn, self.cancel_btn, self.close_btn):
            buttons.addWidget(b)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.summary)
        lay.addWidget(self.current)
        lay.addWidget(self.overall)
        lay.addWidget(self.tile_bar)
        lay.addWidget(self.log, 1)
        lay.addLayout(buttons)

        for w in (self.preset, self.format):
            w.currentIndexChanged.connect(lambda _: self._update_summary())
        for w in (self.in_dir, self.out_dir, self.suffix):
            w.textChanged.connect(lambda _: self._update_summary())
        self.recursive.toggled.connect(lambda _: self._update_summary())

        self.thread = QThread(self)
        self.worker = BatchWorker()
        self.worker.moveToThread(self.thread)
        self.start.connect(self.worker.run)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.stage.connect(self.current.setText)
        self.worker.tiles.connect(self.on_tiles)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.finished.connect(self.on_finished)
        self.thread.start()
        self.counts = {"done": 0, "skipped": 0, "failed": 0}
        self._set_running(False)
        self._update_summary()

    def _pick(self, edit: QLineEdit, title: str):
        path = QFileDialog.getExistingDirectory(self, title, edit.text() or str(Path.home()))
        if path:
            edit.setText(path)

    def _settings(self) -> dict:
        name = self.preset.currentData()
        return presets.load(name) if name else self.win.settings()

    def _plan(self) -> dict:
        """What would run with the current choices; raises ValueError with a reason."""
        in_dir = Path(self.in_dir.text()).expanduser()
        out_dir = Path(self.out_dir.text()).expanduser()
        if not in_dir.is_dir():
            raise ValueError("The input folder doesn't exist.")
        if not self.out_dir.text().strip():
            raise ValueError("Pick an output folder.")
        job = batch.job_from_settings(self._settings(), {k: SLIDER_DEFAULTS[k] / 100 for k in ENH_KEYS})
        photos = batch.find_photos(in_dir, self.recursive.isChecked())
        if out_dir.resolve() != in_dir.resolve() and out_dir.resolve().is_relative_to(in_dir.resolve()):
            photos = [p for p in photos if not p.resolve().is_relative_to(out_dir.resolve())]  # not our own output
        if not photos:
            raise ValueError("No photos in the input folder.")
        return {"photos": photos, "job": job, "in_dir": in_dir, "out_dir": out_dir,
                "ext": batch.FORMATS[self.format.currentText()], "suffix": self.suffix.text().strip(),
                "overwrite": self.overwrite.isChecked()}

    def _update_summary(self):
        if self.running:
            return
        try:
            plan = self._plan()
        except Exception as e:
            self.summary.setText(str(e))
            self.start_btn.setEnabled(False)
            return
        job = plan["job"]
        steps = [s for s, on in (("AI denoise", job.denoise), ("AI sharpen", job.sharpen)) if on]
        ai = " + ".join(steps) if steps else "no AI steps"
        slow = "" if not steps or engine._device().type != "cpu" else " On the CPU the AI steps take a long time per photo."
        n = len(plan["photos"])
        self.summary.setText(f"{n} photo{'s' if n != 1 else ''} · {ai} · {self.format.currentText()}.{slow}")
        self.start_btn.setEnabled(True)

    def _set_running(self, on: bool):
        self.running = on
        for w in (self.preset, self.in_dir, self.out_dir, self.format, self.suffix, self.recursive, self.overwrite,
                  *self.browse):
            w.setEnabled(not on)
        self.start_btn.setVisible(not on)
        self.cancel_btn.setVisible(on)
        self.cancel_btn.setEnabled(on)
        self.close_btn.setEnabled(not on)

    def start_batch(self):
        try:
            plan = self._plan()
        except Exception as e:
            QMessageBox.warning(self, "Batch", str(e))
            return
        if self.win.busy:  # don't share the GPU with an interactive AI run
            self.win.cancel_job()
        self.log.clear()
        self.counts = {"done": 0, "skipped": 0, "failed": 0}
        self.overall.setRange(0, len(plan["photos"]))
        self.overall.setValue(0)
        self.tile_bar.setVisible(bool(plan["job"].denoise or plan["job"].sharpen))  # AI tiles only
        self.worker.cancel = False
        self._set_running(True)
        self.start.emit(plan)

    def cancel_batch(self):
        self.worker.cancel = True
        self.cancel_btn.setEnabled(False)
        self.current.setText("Cancelling after the current tile…")

    def on_file_started(self, i: int, n: int, name: str):
        self.overall.setValue(i)
        self.overall.setFormat(f"photo {i + 1}/{n}")
        self.tile_bar.setRange(0, 1)
        self.tile_bar.setValue(0)

    def on_tiles(self, done: int, total: int):
        self.tile_bar.setRange(0, total)
        self.tile_bar.setValue(done)

    def on_file_done(self, status: str, msg: str):
        self.counts[status] += 1
        self.log.addItem({"done": "✓ ", "skipped": "– ", "failed": "✗ "}[status] + msg)
        self.log.scrollToBottom()
        self.overall.setValue(self.overall.value() + 1)

    def on_finished(self, complete: bool):
        self._set_running(False)
        self.overall.setValue(self.overall.maximum() if complete else self.overall.value())
        self.overall.setFormat("%v/%m photos")
        c = self.counts
        head = "Done" if complete else "Cancelled"
        self.current.setText(f"{head}: {c['done']} saved, {c['skipped']} skipped, {c['failed']} failed.")
        self.win.status.setText(f"Batch {head.lower()}: {c['done']} saved")

    def closeEvent(self, e):
        if self.running:  # cancel first
            e.ignore()
            return
        super().closeEvent(e)

    def done(self, result: int):
        """Every way of closing (Close, Esc, the window's ×) ends here."""
        if self.running:
            return
        self.thread.quit()
        self.thread.wait(5000)
        super().done(result)


class MainWindow(QMainWindow):
    start_job = Signal(object)
    start_load = Signal(int, object)
    start_save = Signal(object)

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
        open_btn.clicked.connect(self.open_dialog)
        self.save_btn = QPushButton("Save…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_dialog)

        self.preset_box = QComboBox()
        self.preset_box.setMinimumWidth(180)
        self.preset_box.activated.connect(self.preset_chosen)
        save_preset = QPushButton("Save preset…")
        save_preset.clicked.connect(self.save_preset)
        del_preset = QPushButton("Delete preset")
        del_preset.clicked.connect(self.delete_preset)
        batch_btn = QPushButton("Batch…")
        batch_btn.setToolTip("Process a whole folder with a preset")
        batch_btn.clicked.connect(lambda: BatchDialog(self).exec())
        self.refresh_presets()

        self.crop_btn = QPushButton("Crop")
        self.crop_btn.setCheckable(True)
        self.crop_btn.setShortcut("C")
        self.crop_btn.setToolTip("Crop (C). Drag the frame or its handles; Enter = done, Esc = undo changes")
        self.crop_btn.setEnabled(False)
        self.crop_btn.toggled.connect(self.crop_toggled)
        self.aspect_box = QComboBox()
        for name, ratio in CROP_ASPECTS:
            self.aspect_box.addItem(name, ratio)
        self.aspect_box.setToolTip("Crop aspect ratio")
        self.aspect_box.currentIndexChanged.connect(lambda _: self.aspect_changed())
        self.aspect_flip = QPushButton("⇄")
        self.aspect_flip.setToolTip("Swap portrait / landscape")
        self.aspect_flip.setFixedWidth(32)
        self.aspect_flip.clicked.connect(self.flip_aspect)
        self.crop_reset = QPushButton("Reset crop")
        self.crop_reset.clicked.connect(lambda: self.view.set_crop(None))
        self.portrait = False
        self._crop_before: tuple | None = None  # crop when crop mode started (Esc restores it)
        self.view.crop_changed.connect(self.on_crop_changed)
        self.view.crop_finished.connect(self.crop_finished)
        self._set_crop_controls()

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setTextVisible(True)
        self.progress.hide()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Stop the AI denoise / sharpen run (after the current tile)")
        self.cancel_btn.clicked.connect(self.cancel_job)
        self.cancel_btn.hide()
        self.load_req = 0  # id of the latest open request; older results are dropped
        self.load_text: str | None = None  # overlay text while a photo is loading
        self.status = QLabel(engine.device_name())

        # Library / Edit switch
        self.library_btn = QPushButton("Library")
        self.edit_btn = QPushButton("Edit")
        modes = QButtonGroup(self)
        for b, tip in ((self.library_btn, "Browse folders and photos (G)"), (self.edit_btn, "Develop and enhance (E)")):
            b.setCheckable(True)
            b.setToolTip(tip)
            modes.addButton(b)
        self.library_btn.clicked.connect(lambda: self.set_mode(edit=False))
        self.edit_btn.clicked.connect(lambda: self.set_mode(edit=True))

        preset_label = QLabel("   Preset:")
        crop_gap = QLabel("   ")
        # shown only in Edit mode (hidden widgets' shortcuts, like C for crop, don't fire)
        self.edit_only = [self.save_btn, preset_label, self.preset_box, save_preset, del_preset,
                          crop_gap, self.crop_btn, self.aspect_box, self.aspect_flip, self.crop_reset]
        top = QHBoxLayout()
        for w in (self.library_btn, self.edit_btn, QLabel("  "), open_btn, self.save_btn, preset_label,
                  self.preset_box, save_preset, del_preset, batch_btn, crop_gap, self.crop_btn,
                  self.aspect_box, self.aspect_flip, self.crop_reset):
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
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.view, 1)
        body.addWidget(side_w)
        editor = QWidget()
        editor.setLayout(body)

        self.library = LibraryView()
        self.library.open_photo.connect(self.open_from_library)
        self.pages = QStackedWidget()
        self.pages.addWidget(self.library)
        self.pages.addWidget(editor)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addWidget(self.pages, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self._build_menu(batch_btn)

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

        self.load_thread = QThread(self)
        self.loader = Loader()
        self.loader.moveToThread(self.load_thread)
        self.start_load.connect(self.loader.load)
        self.loader.loaded.connect(self.on_loaded)
        self.loader.failed.connect(self.on_load_failed)
        self.saver = Saver()  # shares the loader's thread: file work, one at a time
        self.saver.moveToThread(self.load_thread)
        self.start_save.connect(self.saver.save)
        self.saver.progress.connect(self.on_save_progress)
        self.saver.saved.connect(self.on_saved)
        self.saver.failed.connect(self.on_save_failed)
        self.saving = False
        self.load_thread.start()

        self.refresh = QTimer(self)
        self.refresh.setInterval(400)
        self.refresh.timeout.connect(self.view.ai_updated)

        self._sync_enh_view()
        self.set_mode(edit=False)  # start in the Library
        if self.denoise_box.count() == 0:
            self.status.setText("No models in models/ — run setup/download-models.sh")

    # --- menu and Library / Edit ----------------------------------------
    def _build_menu(self, batch_btn: QPushButton):
        def action(menu, text, slot, shortcut=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            menu.addAction(a)
            return a

        file_menu = self.menuBar().addMenu("&File")
        action(file_menu, "&Open photo…", self.open_dialog, "Ctrl+O")
        action(file_menu, "Open &folder in Library…", self.open_folder_dialog, "Ctrl+Shift+O")
        self.save_action = action(file_menu, "&Save…", lambda: self.save_btn.isEnabled() and self.save_dialog(), "Ctrl+S")
        file_menu.addSeparator()
        action(file_menu, "&Batch…", batch_btn.click)
        file_menu.addSeparator()
        action(file_menu, "&Quit", self.close, "Ctrl+Q")
        view_menu = self.menuBar().addMenu("&View")
        action(view_menu, "&Library", lambda: self.set_mode(edit=False), "G")
        action(view_menu, "&Edit", lambda: self.set_mode(edit=True), "E")

    def set_mode(self, edit: bool):
        self.pages.setCurrentIndex(1 if edit else 0)
        (self.edit_btn if edit else self.library_btn).setChecked(True)
        for w in self.edit_only:
            w.setVisible(edit)
        if edit:
            self.view.setFocus()
        else:
            if self.crop_btn.isChecked():
                self.crop_btn.setChecked(False)
            self.library.grid.setFocus()

    def open_folder_dialog(self):
        start = str(self.library.folder or Path.home() / "Pictures")
        folder = QFileDialog.getExistingDirectory(self, "Open folder", start)
        if folder:
            self.library.show_folder(Path(folder))
            self.set_mode(edit=False)

    def open_from_library(self, path: Path):
        if self.photo is not None and path == self.photo.path:
            self.set_mode(edit=True)  # already open: keep its edits and AI results
        else:
            self.open_path(path)

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

    def on_rendered(self, rgb: np.ndarray, hist: tuple, frac: float):
        note = ""
        if self.photo is not None and self.photo.meta.get("raw") and self.view.mode == "develop":
            if frac >= 0.002:
                if self.dev_params["highlights"] < 0:
                    note = f"Highlights is recovering RAW detail past white ({frac * 100:.1f}%)"
                else:
                    note = f"RAW has detail past white ({frac * 100:.1f}%): pull Highlights left"
        self.hist.set_histogram(hist, note)

    # --- crop -------------------------------------------------------------
    def _set_crop_controls(self):
        on = self.crop_btn.isChecked()
        for w in (self.aspect_box, self.aspect_flip):
            w.setVisible(on)
        self.crop_reset.setVisible(on or self.view.crop is not None)

    def crop_toggled(self, on: bool):
        if on:
            self._crop_before = self.view.crop
            self.aspect_changed()
        self.view.set_crop_mode(on)
        self.view.setFocus()  # for Enter / Esc
        self._set_crop_controls()
        self.on_crop_changed(self.view.crop)

    def crop_finished(self, keep: bool):
        if not keep:
            self.view.set_crop(self._crop_before)
        self.crop_btn.setChecked(False)

    def _aspect(self) -> float | None:
        ratio = self.aspect_box.currentData()
        if ratio == "original" and self.photo is not None:
            h, w, _ = self.photo.linear.shape
            ratio = w / h
        if not isinstance(ratio, float):
            return None
        return 1 / ratio if self.portrait else ratio

    def aspect_changed(self):
        if self.crop_btn.isChecked():
            self.view.set_crop_aspect(self._aspect())

    def flip_aspect(self):
        self.portrait = not self.portrait
        self.aspect_changed()

    def on_crop_changed(self, crop):
        self._set_crop_controls()
        if self.photo is None:
            return
        h, w, _ = self.photo.linear.shape
        cw, ch = (crop[2] - crop[0], crop[3] - crop[1]) if crop else (w, h)
        if self.crop_btn.isChecked() or crop:
            self.status.setText(f"Crop {cw}×{ch}" + ("" if self.crop_btn.isChecked() else " (C to change)"))

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

    @property
    def loading(self) -> bool:
        return self.load_text is not None

    def open_path(self, path: Path):
        """Start loading a photo (in the editor); on_loaded takes over when it's decoded."""
        self.set_mode(edit=True)
        self.load_req += 1
        self.load_text = f"Loading {path.name}…"
        if not self.saving:  # else it shows once the save is done (same thread)
            self.view.set_loading(self.load_text)
        self.start_load.emit(self.load_req, path)

    def on_load_failed(self, req: int, name: str, msg: str):
        if req != self.load_req:
            return
        self.load_text = None
        if not self.saving:
            self.view.set_loading(None)
        QMessageBox.warning(self, "Could not open", f"{name}: {msg}")

    def on_loaded(self, req: int, photo: imageio.Photo, g: float):
        if req != self.load_req:
            return  # another photo was opened meanwhile
        self.load_text = None
        if not self.saving:
            self.view.set_loading(None)
        path = photo.path
        if self.busy:
            self.worker.cancel = True
            self.pending = True
        self.photo = photo
        self.photo_gen += 1
        self.g = g
        self.library.show_folder(path.parent, select=path)
        self.enc_orig = None
        self.results.clear()
        self.running.clear()
        self.started.clear()
        self.save_target = None
        self.save_btn.setEnabled(True)
        h, w, _ = photo.linear.shape
        kind = f"RAW, base exposure +{photo.baseline_ev:.2f} EV" if photo.meta.get("raw") else f"{photo.meta.get('bits')}-bit"
        self.setWindowTitle(f"Freeroom — {path.name} ({w}×{h}, {kind})")
        self.crop_btn.setEnabled(True)
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
        if self.photo is None or self.saving:
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
        """Hand the full-size develop + write to the saver thread, with the
        settings as they are now (changing sliders meanwhile doesn't affect it)."""
        out, self.save_target = self.save_target, None
        dn_key, sh_key = self._keys()
        self.saving = True
        self.save_btn.setEnabled(False)
        self.view.set_progress(f"Saving {out.name}…", 0)
        self.start_save.emit({
            "photo": self.photo, "out": out, "g": self.g,
            "dn": self.results.get(dn_key), "sharp": self.results.get(sh_key),
            "dev": dict(self.dev_params), "enh": self.effective_enh(), "crop": self.view.crop,
        })

    def on_save_progress(self, done: int, total: int, label: str):
        self.view.set_progress(f"Saving: {label}…", round(100 * done / total))

    def _save_ended(self):
        self.saving = False
        self.save_btn.setEnabled(self.photo is not None)
        self.view.set_loading(self.load_text)  # a photo opened meanwhile, or None

    def on_saved(self, path: str):
        self._save_ended()
        self.status.setText(f"Saved {path}")

    def on_save_failed(self, msg: str):
        self._save_ended()
        QMessageBox.warning(self, "Could not save", msg)

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
        self.load_thread.quit()
        self.load_thread.wait(5000)
        self.view.shutdown()
        self.library.shutdown()
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
