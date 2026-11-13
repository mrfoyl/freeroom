"""Freeroom GUI: open a photo, develop it, enhance it with AI, save.

Raw tab: instant, per-pixel dials on the linear photo, then an optional HDR
section (local tone mapping) and a look (colour LUT, see looks.py).
B&W tab: more development, applied after the Raw tab.
Enhancement tab: AI denoise + sharpen, shown on top of the development.
Crop (top bar): applied to the view and the saved file; kept per photo, not in presets.
Batch… (top bar): one preset over a whole folder (see batch.py).
Library (G) / Edit (E): browse folders with thumbnails and a preview (browser.py),
open a photo in the editor.
Keep edits with photo (Ctrl+K): a sidecar file next to it (sidecar.py), read
back when the photo is opened again.
Each AI model runs at most once per photo; its result is kept and the dials
are applied on top (see pipeline.py), so only a new photo re-runs the AI.
"""

from __future__ import annotations

import copy
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QToolButton,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import batch, develop, engine, imageio, layers, looks, pipeline, presets, redeye, segment, sidecar
from .browser import LibraryView
from .widgets import Histogram, PhotoView, ResetSlider

SAVE_DIR = Path.home() / "Pictures" / "freeroom"
ICON = Path(__file__).resolve().parent / "assets" / "freeroom.png"  # optional, gitignored
SIDE_MIN, SIDE_DEFAULT, SIDE_MAX = 330, 400, 720  # side panel width (px), drag its edge
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
# Colour mixer: per colour, Hue (±30°), Saturation and Luminance; one page each
HSL_PAGES = [("hue", "Hue", lambda v: f"{v * develop.HSL_HUE / 100:+.0f}°"),
             ("sat", "Saturation", lambda v: f"{v:+d}"),
             ("lum", "Luminance", lambda v: f"{v:+d}")]
DEV_HSL = {
    part: [(f"hsl_{name}_{part}", f"<span style='color:{BW_SWATCH[name]}'>■</span> {name.capitalize()}",
            -100, 100, 0, fmt) for name, _ in develop.BW_COLOURS]
    for part, _, fmt in HSL_PAGES
}
PCT = lambda v: f"{v}%"  # noqa: E731
DEV_LOOK = [("lut_strength", "Strength", 0, 100, 100, PCT)]
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
LOOK_KEYS = [s[0] for s in DEV_LOOK]
HSL_KEYS = [s[0] for specs in DEV_HSL.values() for s in specs]
DEV_KEYS = RAW_KEYS + HSL_KEYS + HDR_KEYS + BW_KEYS + LOOK_KEYS
ENH_KEYS = [s[0] for s in ENH_NR + ENH_SH]
SLIDER_DEFAULTS = {s[0]: s[4] for s in DEV_LIGHT + DEV_COLOUR + DEV_HDR + DEV_BW + DEV_LOOK + ENH_NR + ENH_SH
                   + [spec for specs in DEV_HSL.values() for spec in specs]}

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
        # Result keys the settings still need, replaced by the GUI thread when they
        # change: a stage outside it is skipped, or stopped after its current tile.
        self.wanted: frozenset = frozenset()
        self.ran = 0  # stages completed in the last job

    def _wanted(self, key) -> bool:
        return key in self.wanted

    def _model(self, path: Path) -> engine.Restorer:
        if path not in self.models:
            self.status.emit(f"Loading model {path.stem}…")
            self.models[path] = engine.Restorer(path)
        return self.models[path]

    def run(self, stages: list[tuple]):
        """Run (key, label, model, src, dst) stages in order. dst starts as a copy of src."""
        try:
            t0 = time.monotonic()
            self.ran = 0
            for key, label, path, src, dst in stages:
                if not self._wanted(key):
                    continue  # switched off before it started
                model = self._model(path)
                dst[:] = src
                self.stage.emit(key)
                self.status.emit(f"{label} with {path.stem}…")
                stop = lambda: self.cancel or not self._wanted(key)  # noqa: E731
                if not model.run(src, dst, progress=self.progress.emit, cancelled=stop):
                    if self.cancel:
                        self.finished.emit(False, 0.0)
                        return
                    continue  # switched off while running: drop it, go on with the rest
                if label == "Sharpening":
                    engine.keep_detail(src, dst)
                self.stage_done.emit(key)
                self.ran += 1
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

            total += 2 * len(job.get("layers", []))
            full_lin = job["photo"].linear
            lin, dn, sharp, base = job["photo"].linear, job["dn"], job["sharp"], None
            spots = job["spots"]
            if spots is None:  # saved before the search for red eyes had finished
                spots = redeye.detect(lin, job["photo"].baseline_ev)
            if job["crop"]:
                # develop only the cropped part; HDR still uses the whole photo's map
                x0, y0, x1, y1 = job["crop"]
                if job["dev"].get("hdr"):
                    base = develop.hdr_base(develop.hdr_map(lin), lin.shape, (y1 - y0, x1 - x0), (y0, x0, y1, x1))
                cut = np.s_[y0:y1, x0:x1]
                lin = lin[cut]
                dn = None if dn is None else dn[cut]
                sharp = None if sharp is None else sharp[cut]
            fix = lambda x: redeye.apply(x, spots, (y0, x0, y1, x1) if job["crop"] else None)  # noqa: E731
            rgb = pipeline.compose(
                lin, dn, sharp, job["g"], job["dev"], job["photo"].baseline_ev, job["enh"],
                base=base, progress=step, fix=fix,
            )
            H, W = full_lin.shape[:2]
            region = (y0, x0, y1, x1) if job["crop"] else (0, 0, H, W)
            for n, d in enumerate(job.get("layers", []), 1):
                step(f"Developing layer {n}")
                cut = np.s_[region[0]:region[2], region[1]:region[3]]
                lbase = None
                if d["dev"].get("hdr") and job["crop"]:
                    lbase = develop.hdr_base(develop.hdr_map(full_lin), full_lin.shape, lin.shape[:2], region)
                above = pipeline.compose(
                    lin, None if d["dn"] is None else d["dn"][cut], None if d["sharp"] is None else d["sharp"][cut],
                    job["g"], d["dev"], job["photo"].baseline_ev, d["enh"], base=lbase, fix=fix,
                )
                step(f"Blending layer {n}")
                rgb = layers.blend(rgb, above, layers.region_mask(d["mask"], (H, W), region, lin.shape[:2]))
            step(f"Writing {job['out'].name}")
            saved = imageio.save(job["photo"], rgb, job["out"], **job.get("options", {}))
            self.saved.emit(str(saved))
        except Exception as e:
            self.failed.emit(str(e))


class RedEyeFinder(QObject):
    """Looks through a whole photo for red eyes, off the GUI thread."""

    found = Signal(int, tuple)  # photo generation, spots
    failed = Signal(int, str)

    def find(self, gen: int, linear: np.ndarray, baseline_ev: float):
        try:
            self.found.emit(gen, tuple(redeye.detect(linear, baseline_ev)))
        except Exception as e:
            self.failed.emit(gen, str(e))


class SegWorker(QObject):
    """AI selection (SAM) off the GUI thread: reads a photo once, then a mask per prompt set."""

    ready = Signal(int)  # photo generation
    masked = Signal(int, int, object)  # photo generation, layer id, probability mask (work size)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.seg: segment.Segmenter | None = None
        self.gen = -1

    def prepare(self, gen: int, linear_small: np.ndarray, baseline_ev: float):
        try:
            if self.seg is None:
                self.seg = segment.Segmenter()
            self.seg.set_image(develop.develop(linear_small, develop.DEFAULTS, baseline_ev))
            self.gen = gen
            self.ready.emit(gen)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    def predict(self, gen: int, uid: int, points: list, labels: list):
        if gen != self.gen:
            return  # for a photo that has been closed
        try:
            self.masked.emit(gen, uid, self.seg.predict(points, labels))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


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
                                            self.tiles.emit, lambda: self.cancel, plan["options"])
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
                "ext": batch.FORMATS[self.format.currentText()][0], "options": batch.FORMATS[self.format.currentText()][1],
                "suffix": self.suffix.text().strip(),
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
    start_find = Signal(int, object, float)
    start_seg = Signal(int, object, float)
    start_mask = Signal(int, int, object, object)

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
        self.save_options: dict = {}  # imageio.save options of the format picked in the save dialog
        self._applying = False  # loading a preset: hold back pipeline runs
        # layers above the base; the tabs edit the active one (None = Base)
        self.layers: list[layers.Layer] = []
        self.active: int | None = None
        self.base_settings: dict | None = None  # the Base's settings while a layer is edited
        self._switching = False  # loading a layer's settings into the tabs
        self.seg_gen = -1  # photo the AI selection has read (or is reading)

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
        self.keep_btn = QPushButton("Keep edits")
        self.keep_btn.setToolTip(
            "Keep the edits with the photo (Ctrl+K): settings, look, crop and layers go in a small "
            "file next to it (photo name + .freeroom.json) and come back when you open it again"
        )
        self.keep_btn.setEnabled(False)
        self.keep_btn.clicked.connect(self.keep_edits)

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
        self.cancel_btn.setToolTip("Stop the AI step that's running and untick its section")
        self.cancel_btn.clicked.connect(self.stop_enhancement)
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
        self.edit_only = [self.save_btn, self.keep_btn, preset_label, self.preset_box, save_preset, del_preset,
                          crop_gap, self.crop_btn, self.aspect_box, self.aspect_flip, self.crop_reset]
        top = QHBoxLayout()
        for w in (self.library_btn, self.edit_btn, QLabel("  "), open_btn, self.save_btn, self.keep_btn, preset_label,
                  self.preset_box, save_preset, del_preset, batch_btn, crop_gap, self.crop_btn,
                  self.aspect_box, self.aspect_flip, self.crop_reset):
            top.addWidget(w)
        # long texts (a saved path…) get cut off instead of widening the window
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.status, 1)
        top.addWidget(self.progress)
        top.addWidget(self.cancel_btn)

        # --- tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_develop_tab(), "Raw")
        self.tabs.addTab(self._build_bw_tab(), "B&&W")
        self.enh_tab = self._build_enhance_tab()
        self.tabs.addTab(self.enh_tab, "Enhancement")
        self.tabs.currentChanged.connect(self.tab_changed)

        # one line, the whole story on hover: the dials get the height
        hint = QLabel("<b>Loupe</b> 100% · Hold mouse / Space: before · Scroll: loupe size")
        hint.setToolTip(
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
        side.addWidget(self._build_layers_box())
        side.addWidget(self.tabs, 1)
        side.addWidget(hint)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setMinimumWidth(SIDE_MIN)
        side_w.setMaximumWidth(SIDE_MAX)
        side_w.setStyleSheet(GROUP_STYLE)

        # the side panel's edge can be dragged; its width is remembered
        self.ui_settings = QSettings("freeroom", "freeroom")
        editor = QSplitter(Qt.Orientation.Horizontal)
        editor.addWidget(self.view)
        editor.addWidget(side_w)
        editor.setChildrenCollapsible(False)
        editor.setStretchFactor(0, 1)
        editor.setStretchFactor(1, 0)
        editor.splitterMoved.connect(lambda *_: self.ui_settings.setValue("side_width", editor.sizes()[1]))
        self.editor_split = editor
        self._side_sized = False  # the saved width is set once the editor is first laid out

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
        self.finder = RedEyeFinder()  # same thread: it reads the photo, like the others
        self.finder.moveToThread(self.load_thread)
        self.start_find.connect(self.finder.find)
        self.finder.found.connect(self.on_redeye_found)
        self.finder.failed.connect(self.on_redeye_failed)
        self.saving = False
        self.load_thread.start()

        self.seg_thread = QThread(self)
        self.segworker = SegWorker()
        self.segworker.moveToThread(self.seg_thread)
        self.start_seg.connect(self.segworker.prepare)
        self.start_mask.connect(self.segworker.predict)
        self.segworker.ready.connect(self.on_seg_ready)
        self.segworker.masked.connect(self.on_masked)
        self.segworker.failed.connect(self.on_seg_failed)
        self.seg_thread.start()
        self.view.mask_clicked.connect(self.on_mask_click)
        self.view.mask_stroked.connect(self.on_mask_stroke)
        self.view.mask_finished.connect(lambda: self._set_mask_tool(None))
        QShortcut(QKeySequence.StandardKey.Undo, self.view, self.mask_undo)

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

        # in the window, also where the desktop would move menus elsewhere (KDE global menu)
        self.menuBar().setNativeMenuBar(False)
        file_menu = self.menuBar().addMenu("&File")
        action(file_menu, "&Open photo…", self.open_dialog, "Ctrl+O")
        action(file_menu, "Open &folder in Library…", self.open_folder_dialog, "Ctrl+Shift+O")
        self.save_action = action(file_menu, "&Save…", lambda: self.save_btn.isEnabled() and self.save_dialog(), "Ctrl+S")
        self.keep_action = action(file_menu, "&Keep edits with photo", self.keep_edits, "Ctrl+K")
        self.keep_action.setToolTip("Write the edits to a small file next to the photo; they come back when you open it again")
        self.keep_action.setEnabled(False)
        file_menu.addSeparator()
        action(file_menu, "&Batch…", batch_btn.click)
        file_menu.addSeparator()
        action(file_menu, "&Quit", self.close, "Ctrl+Q")
        view_menu = self.menuBar().addMenu("&View")
        action(view_menu, "&Library", lambda: self.set_mode(edit=False), "G")
        action(view_menu, "&Edit", lambda: self.set_mode(edit=True), "E")

    def set_mode(self, edit: bool):
        self.pages.setCurrentIndex(1 if edit else 0)
        if edit and not self._side_sized:
            QTimer.singleShot(0, self._size_side)
        (self.edit_btn if edit else self.library_btn).setChecked(True)
        for w in self.edit_only:
            w.setVisible(edit)
        if edit:  # these show only when they apply
            self._set_crop_controls()
        if edit:
            self.view.setFocus()
        else:
            if self.crop_btn.isChecked():
                self.crop_btn.setChecked(False)
            self.redeye_btn.setChecked(False)
            self.library.grid.setFocus()

    def _size_side(self):
        """The side panel at its remembered width (the splitter has a real size now)."""
        total = sum(self.editor_split.sizes())
        if self._side_sized or total <= 0 or not self.editor_split.isVisible():
            return
        self._side_sized = True
        try:
            side = int(self.ui_settings.value("side_width", SIDE_DEFAULT))
        except (TypeError, ValueError):
            side = SIDE_DEFAULT
        side = min(max(side, SIDE_MIN), SIDE_MAX)
        self.editor_split.setSizes([max(1, total - side), side])

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
        form.setVerticalSpacing(2)  # many dials: keep them close, the panel is short
        row = QHBoxLayout()
        row.addWidget(s)
        row.addWidget(val)
        form.addRow(text, row)
        self.sliders[key] = (s, val)

    def _build_develop_tab(self) -> QWidget:
        # Shooting info of the open photo (filled in by _show_shot)
        shot = QGroupBox("Photo")
        shot_lay = QVBoxLayout(shot)
        self.shot_head = QLabel("No photo open")
        self.shot_head.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self.shot_head.setWordWrap(True)
        self.shot_rows = QLabel()
        self.shot_rows.setWordWrap(True)
        self.shot_rows.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.shot_rows.hide()
        shot_lay.addWidget(self.shot_head)
        shot_lay.addWidget(self.shot_rows)

        light = QGroupBox("Light")
        form = QFormLayout(light)
        for spec in DEV_LIGHT:
            self._add_slider(form, spec, self.dev_changed)
        colour = QGroupBox("Colour")
        form = QFormLayout(colour)
        for spec in DEV_COLOUR:
            self._add_slider(form, spec, self.dev_changed)
        mixer = self._build_hsl_section()
        self.hdr_group = self._dev_group("HDR tone mapping", "hdr")
        self.hdr_group.setToolTip(
            "Squeezes a high-contrast scene into the picture: Compression lifts shadows and "
            "pulls highlights down area by area, Local detail brings back (or tames) texture "
            "and local contrast. Works best on RAW files."
        )
        form = QFormLayout(self.hdr_group)
        for spec in DEV_HDR:
            self._add_slider(form, spec, self.dev_changed)
        look = self._build_look_group()
        reset = QPushButton("Reset development")
        reset.clicked.connect(self.reset_development)
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.addWidget(shot)
        lay.addWidget(light)
        lay.addWidget(colour)
        lay.addWidget(mixer)
        lay.addWidget(self.hdr_group)
        lay.addWidget(look)
        lay.addWidget(reset)
        lay.addStretch(1)
        scroll = QScrollArea()  # so the dials stay reachable on small screens
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(tab)
        return scroll

    def _build_hsl_section(self) -> QWidget:
        """Colour mixer (hue / saturation / luminance per colour), folded at start."""
        self.hsl_toggle = QToolButton()
        self.hsl_toggle.setCheckable(True)
        self.hsl_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.hsl_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.hsl_toggle.setAutoRaise(True)
        self.hsl_toggle.setStyleSheet("QToolButton { font-weight: bold; font-size: 11pt; border: none; }")
        self.hsl_toggle.setToolTip("Turn, strengthen or brighten single colours: e.g. bluer sky, greener grass, warmer skin")
        body = QGroupBox()
        body.setVisible(False)
        self.hsl_body = body
        pages = QStackedWidget()
        pick = QHBoxLayout()
        pick.setSpacing(0)
        group = QButtonGroup(body)
        for i, (part, title, _) in enumerate(HSL_PAGES):
            page = QWidget()
            form = QFormLayout(page)
            form.setContentsMargins(0, 4, 0, 0)
            for spec in DEV_HSL[part]:
                self._add_slider(form, spec, self.dev_changed)
            pages.addWidget(page)
            b = QPushButton(title)
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.clicked.connect(lambda _=False, i=i: pages.setCurrentIndex(i))
            group.addButton(b)
            pick.addWidget(b)
        self.hsl_pages = pages
        reset = QPushButton("Reset colour mixer")
        reset.clicked.connect(lambda: self._reset(HSL_KEYS))
        lay = QVBoxLayout(body)
        lay.addLayout(pick)
        lay.addWidget(pages)
        lay.addWidget(reset)
        self.hsl_toggle.toggled.connect(self._fold_hsl)
        self._hsl_title()
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 6, 0, 0)
        outer.setSpacing(2)
        outer.addWidget(self.hsl_toggle)
        outer.addWidget(body)
        return panel

    def _fold_hsl(self, open_: bool):
        self.hsl_toggle.setArrowType(Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow)
        self.hsl_body.setVisible(open_)

    def _hsl_title(self):
        used = any(self.dev_params.get(k) for k in HSL_KEYS)  # so a folded mixer still says it's on
        self.hsl_toggle.setText("Colour mixer" + ("  ·  in use" if used else ""))

    def _build_look_group(self) -> QGroupBox:
        """Look: a colour LUT (.cube or Hald CLUT) over the developed photo."""
        group = QGroupBox("Look (LUT)")
        group.setToolTip(
            "A colour look from a LUT file (.cube, or a Hald CLUT .png/.tif), applied after "
            f"everything else. Put LUTs in {looks.LUT_DIR} to have them in the list."
        )
        win = self

        class LutBox(QComboBox):
            def showPopup(self):  # pick up files added to the folder meanwhile
                win.refresh_luts()
                super().showPopup()

        self.lut_box = LutBox()
        self.lut_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.lut_box.setMinimumContentsLength(6)
        self.lut_box.currentIndexChanged.connect(self.lut_chosen)
        pick = QToolButton()
        pick.setText("File…")
        pick.setToolTip("Use a LUT file from anywhere")
        pick.clicked.connect(self.pick_lut)
        folder = QToolButton()
        folder.setText("Folder")
        folder.setToolTip(f"Open the LUT folder ({looks.LUT_DIR})")
        folder.clicked.connect(self.open_lut_folder)
        row = QHBoxLayout()
        row.addWidget(self.lut_box, 1)
        row.addWidget(pick)
        row.addWidget(folder)
        form = QFormLayout(group)
        form.addRow("LUT", row)
        for spec in DEV_LOOK:
            self._add_slider(form, spec, self.dev_changed)
        self.lut_note = QLabel()
        self.lut_note.setStyleSheet("color: #e08040;")
        self.lut_note.setWordWrap(True)
        self.lut_note.hide()
        form.addRow(self.lut_note)
        export = QPushButton("Export as LUT…")
        export.setToolTip(
            "Save the Raw and B&W settings shown in the tabs (and the LUT above) as a .cube LUT, "
            "to use the look in other programs. HDR can't go in a LUT, and highlights past "
            "white that Freeroom recovers from a RAW are clipped in a LUT's input."
        )
        export.clicked.connect(self.export_lut)
        self.lut_size_box = QComboBox()
        for size, text in ((33, "33 (standard)"), (65, "65 (finer)")):
            self.lut_size_box.addItem(text, size)
        self.lut_size_box.setToolTip("Steps per colour: 33 suits nearly every program, 65 follows strong curves more closely")
        row = QHBoxLayout()
        row.addWidget(export, 1)
        row.addWidget(self.lut_size_box)
        form.addRow(row)
        self.refresh_luts()
        self.lut_chosen()
        return group

    def refresh_luts(self, select: str | None = None):
        """Fill the LUT list: None, the files in the LUT folder, and the one in use
        if it lives elsewhere (or is missing)."""
        current = self.dev_params.get("lut", "") if select is None else select
        names = looks.names()
        self.lut_box.blockSignals(True)
        self.lut_box.clear()
        self.lut_box.addItem("None", "")
        for n in names:
            self.lut_box.addItem(looks.label(n), n)
            self.lut_box.setItemData(self.lut_box.count() - 1, n, Qt.ItemDataRole.ToolTipRole)
        if current and current not in names:
            self.lut_box.addItem(looks.label(current), current)
            self.lut_box.setItemData(self.lut_box.count() - 1, current, Qt.ItemDataRole.ToolTipRole)
        self.lut_box.setCurrentIndex(max(0, self.lut_box.findData(current)))
        self.lut_box.blockSignals(False)
        if select is not None and self.dev_params.get("lut", "") != select:
            self.lut_chosen()

    def lut_chosen(self, _=None):
        name = self.lut_box.currentData() or ""
        msg = looks.problem(name)
        self.lut_note.setText(msg or "")
        self.lut_note.setVisible(bool(msg))
        self.sliders["lut_strength"][0].setEnabled(bool(name))
        if self.dev_params.get("lut", "") != name:
            self.dev_params["lut"] = name
            self._push_dev()

    def pick_lut(self):
        start = str(looks.LUT_DIR if looks.LUT_DIR.is_dir() else Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Choose a LUT", start,
                                              "LUTs (*.cube *.CUBE *.png *.PNG *.tif *.tiff *.TIF *.TIFF)")
        if not path:
            return
        try:
            looks.read(Path(path))
        except Exception as e:
            QMessageBox.warning(self, "Not a usable LUT", f"{Path(path).name}: {e}")
            return
        self.refresh_luts(select=looks.setting_name(Path(path)))

    def export_lut(self):
        """The tabs' development settings (Base, or the layer being edited) as a .cube."""
        params = dict(self.dev_params)
        name = (self.preset_box.currentData() or "Freeroom look").replace("/", "_")
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "Export look as LUT", str(SAVE_DIR / f"{name}.cube"),
                                              "Cube LUT (*.cube)")
        if not path:
            return
        out = Path(path)
        if out.suffix.lower() != ".cube":
            out = out.with_name(out.name + ".cube")
        try:
            looks.write_cube(out, develop.look_table(params, self.lut_size_box.currentData()), out.stem)
        except Exception as e:
            QMessageBox.warning(self, "Could not export LUT", f"{out}: {e}")
            return
        left_out = []
        if params.get("hdr"):
            left_out.append("HDR tone mapping")
        if params.get("lut") and looks.problem(params["lut"]):
            left_out.append(f"the LUT “{looks.label(params['lut'])}” (can't be read)")
        self.status.setText(f"Look exported to {out}" + (f" — without {' and '.join(left_out)}" if left_out else ""))

    def open_lut_folder(self):
        looks.LUT_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(looks.LUT_DIR)))

    def _show_shot(self, photo: imageio.Photo):
        head, rows = imageio.describe_shot(photo.meta.get("shot", {}))
        h, w, _ = photo.linear.shape
        rows.append(("Size", f"{w}×{h} ({w * h / 1e6:.1f} MP)"))
        self.shot_head.setText(head or "No exposure data in this file")
        # compact: camera and lens on their own lines, the rest flowing after them,
        # so the dials below get the height; the full table on hover
        grey = "<span style='color:#888'>{}</span>"
        own = [v for k, v in rows if k in ("Camera", "Lens")]
        rest = " · ".join(f"{grey.format(k)} {v}" for k, v in rows if k not in ("Camera", "Lens"))
        self.shot_rows.setText("<br>".join(own + ([rest] if rest else [])))
        self.shot_rows.setToolTip(
            "<table cellspacing='0' cellpadding='1'>"
            + "".join(f"<tr><td style='color:#888; padding-right:8px'>{k}</td><td>{v}</td></tr>" for k, v in rows)
            + "</table>"
        )
        self.shot_rows.show()

    def _dev_group(self, title: str, flag: str) -> QGroupBox:
        """Checkable section that switches a development step (dev_params[flag]) on/off."""
        group = QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(bool(develop.DEFAULTS[flag]))

        def toggled(on):
            self.dev_params[flag] = on
            self._push_dev()

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

    def _build_redeye_group(self) -> QGroupBox:
        group = QGroupBox("Red eye")
        group.setCheckable(True)
        group.setChecked(False)
        group.setToolTip(
            "Looks for red pupils in pairs: same size, as far apart as eyes are for a pupil "
            "that size, on a face the face detector finds. No AI re-run; the dials stay live."
        )
        self.redeye_status = QLabel("Finds red eyes by itself.")
        self.redeye_status.setWordWrap(True)
        self.redeye_btn = QPushButton("Fix by hand")
        self.redeye_btn.setCheckable(True)
        self.redeye_btn.setShortcut("R")
        self.redeye_btn.setToolTip("Click near a pupil a search missed (R); the other eye is found with it. "
                                   "Scroll = search size, right-click = remove a fix")
        self.redeye_btn.toggled.connect(self.redeye_toggled)
        self.redeye_undo = QPushButton("Undo")
        self.redeye_undo.setToolTip("Undo the last fix")
        self.redeye_undo.clicked.connect(lambda: self.view.set_spots(self.view.spots[:-1]))
        self.redeye_again = QPushButton("Find again")
        self.redeye_again.setToolTip("Back to what the search found, without changes by hand")
        self.redeye_again.clicked.connect(self.redeye_find_again)
        buttons = QHBoxLayout()
        for b in (self.redeye_btn, self.redeye_undo, self.redeye_again):
            buttons.addWidget(b)
        lay = QVBoxLayout(group)
        lay.addWidget(self.redeye_status)
        lay.addLayout(buttons)
        group.toggled.connect(self.redeye_section_toggled)
        self.view.redeye_clicked.connect(self.on_redeye_click)
        self.view.spots_changed.connect(self._set_redeye_controls)
        self.view.redeye_finished.connect(lambda: self.redeye_btn.setChecked(False))
        self.redeye_found: tuple | None = None  # what the search found on this photo; None = not yet
        self.redeye_kept: tuple = ()  # the fixes, while the section is off
        self.redeye_searching = False
        self._set_redeye_controls()
        return group

    def _build_enhance_tab(self) -> QWidget:
        self.denoise_box = QComboBox()
        for p in engine.list_models("denoise"):
            self.denoise_box.addItem(engine.pretty_name(p), p)
        self.denoise_box.currentIndexChanged.connect(lambda _: self.enh_settings_changed())
        self.nr_group = QGroupBox("Noise reduction")
        self.nr_group.setCheckable(True)
        self.nr_group.setChecked(False)  # ticking a section starts its AI step
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

        self.re_group = self._build_redeye_group()

        reset = QPushButton("Reset enhancement sliders")
        reset.clicked.connect(lambda: self._reset(ENH_KEYS))
        note = QLabel(
            "Ticking a section starts its AI step; unticking it stops a step that isn't done. "
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
        lay.addWidget(self.re_group)
        lay.addWidget(reset)
        lay.addWidget(note)
        lay.addStretch(1)
        return tab

    # --- layers -------------------------------------------------------------
    def _build_layers_box(self) -> QWidget:
        """The Layers panel, under a header that folds it away (folded at start)."""
        self.layers_toggle = QToolButton()
        self.layers_toggle.setCheckable(True)
        self.layers_toggle.setChecked(False)
        self.layers_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.layers_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.layers_toggle.setAutoRaise(True)
        self.layers_toggle.setStyleSheet("QToolButton { font-weight: bold; border: none; }")
        self.layers_toggle.toggled.connect(self._fold_layers)
        box = QGroupBox()
        self.layers_body = box
        box.setVisible(False)
        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(74)
        self.layer_list.addItem("Base")
        self.layer_list.setCurrentRow(0)
        self.layer_list.currentRowChanged.connect(lambda row: self.set_active(None if row <= 0 else row - 1))
        self.layer_list.itemChanged.connect(self._layer_item_changed)
        self.layer_list.setMouseTracking(True)  # hover a layer: its area shows on the photo
        self.layer_list.itemEntered.connect(self._hover_item)
        self.layer_list.viewport().installEventFilter(self)
        self._hover: layers.Layer | None = None
        self.add_layer_btn = QPushButton("New layer")
        self.add_layer_btn.setToolTip("A layer with its own Raw, B&W and Enhancement settings, inside a selection")
        self.add_layer_btn.clicked.connect(self.add_layer)
        self.copy_layer_btn = QPushButton("Copy")
        self.copy_layer_btn.setToolTip("A copy of this layer: same selection and settings, placed above it")
        self.copy_layer_btn.clicked.connect(self.copy_layer)
        self.del_layer_btn = QPushButton("Delete")
        self.del_layer_btn.clicked.connect(self.delete_layer)
        self.magic_btn = QPushButton("Magic select")
        self.magic_btn.setToolTip("Click an object and the AI selects it; right-click or Ctrl+click: not this")
        self.brush_btn = QPushButton("Brush")
        self.brush_btn.setToolTip("Paint foreground (left) and background (right or Ctrl); the AI follows your strokes")
        for b, tool in ((self.magic_btn, "magic"), (self.brush_btn, "brush")):
            b.setCheckable(True)
            b.clicked.connect(lambda on, t=tool: self._set_mask_tool(t if on else None))
        self.mask_undo_btn = QPushButton("Undo")
        self.mask_undo_btn.clicked.connect(self.mask_undo)
        self.mask_clear_btn = QPushButton("Clear")
        self.mask_clear_btn.clicked.connect(self.mask_clear)
        self.invert_box = QCheckBox("Invert")
        self.invert_box.toggled.connect(lambda on: self._layer_option("invert", on))
        self.show_mask_box = QCheckBox("Show selection")
        self.show_mask_box.toggled.connect(lambda _: self._show_mask())
        self.feather = QSlider(Qt.Orientation.Horizontal)
        self.feather.setRange(0, 100)
        self.feather.setToolTip("Feather: softens the edge of the selection")
        self.feather.valueChanged.connect(lambda v: self._layer_option("feather", v))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(100)
        self.opacity.valueChanged.connect(lambda v: self._layer_option("opacity", v / 100))
        self.layer_note = QLabel("Editing: Base")
        self.layer_note.setStyleSheet("color: #999;")
        self.layer_note.setWordWrap(True)

        lay = QVBoxLayout(box)
        lay.addWidget(self.layer_list)
        row = QHBoxLayout()
        for w in (self.add_layer_btn, self.copy_layer_btn, self.del_layer_btn):
            row.addWidget(w)
        lay.addLayout(row)
        row = QHBoxLayout()
        for w in (self.magic_btn, self.brush_btn, self.mask_undo_btn, self.mask_clear_btn):
            row.addWidget(w)
        lay.addLayout(row)
        form = QFormLayout()
        opts = QHBoxLayout()
        opts.addWidget(self.invert_box)
        opts.addWidget(self.show_mask_box)
        form.addRow(opts)
        form.addRow("Feather", self.feather)
        form.addRow("Opacity", self.opacity)
        lay.addLayout(form)
        lay.addWidget(self.layer_note)
        self._layer_controls()
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        outer.addWidget(self.layers_toggle)
        outer.addWidget(box)
        return panel

    def _fold_layers(self, open_: bool):
        self.layers_toggle.setArrowType(Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow)
        self.layers_body.setVisible(open_)
        if not open_:
            self._hover_item(None)

    def _layer_controls(self):
        L = self.active_layer()
        has_photo = self.photo is not None
        self.add_layer_btn.setEnabled(has_photo)
        for w in (self.del_layer_btn, self.copy_layer_btn, self.magic_btn, self.brush_btn, self.mask_undo_btn, self.mask_clear_btn,
                  self.invert_box, self.show_mask_box, self.feather, self.opacity):
            w.setEnabled(L is not None)
        if L is not None and not segment.available():
            for b in (self.magic_btn, self.brush_btn):
                b.setEnabled(False)
                b.setToolTip("The AI selection model is missing: run setup/download-models.sh")
        n = len(self.layers)
        self.layers_toggle.setText("Layers" + (f" ({n})" if n else "")
                                   + ("" if L is None else f"  ·  editing {L.name}"))
        if L is None:
            self.layer_note.setText("Editing: Base (the whole photo)")
        else:
            self.layer_note.setText(f"Editing: {L.name}. The tabs below change only its selection"
                                    + (" (empty: select something first)." if L.empty() else "."))

    def active_layer(self) -> layers.Layer | None:
        return None if self.active is None else self.layers[self.active]

    def _stack(self, finished_only: bool = False) -> list[dict]:
        """Layers for rendering: visible ones with a selection, bottom first."""
        out = []
        ui = self._ui_settings() if self.active is not None else None
        for i, L in enumerate(self.layers):
            if not L.visible or L.empty():
                continue
            st = ui if i == self.active else L.settings
            dn_key, sh_key = self._keys_of(st)
            get = self.results.get if finished_only else self._layer
            out.append({"mask": L.mask, "dev": dict(st["development"]), "enh": self._enh_of(st),
                        "dn": get(dn_key) if dn_key else None, "sharp": get(sh_key) if sh_key else None})
        return out

    def _push_stack(self):
        if self._switching:
            return
        if self.active is not None:
            self.layers[self.active].settings = self._ui_settings()
        self.view.set_stack(self._stack())

    def add_layer(self):
        if self.photo is None:
            return
        h, w = self.photo.linear.shape[:2]
        n = 1 + max([int(L.name.split()[-1]) for L in self.layers if L.name.split()[-1].isdigit()], default=0)
        # starts as a copy of the Base, so nothing changes until you edit it
        L = layers.Layer(f"Layer {n}", copy.deepcopy(self.settings()), segment.work_shape(h, w), (h, w))
        self.layers.append(L)
        item = QListWidgetItem(L.name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        self.layer_list.addItem(item)
        self.layer_list.setCurrentRow(len(self.layers))  # selects it (set_active)
        if segment.available():
            self.magic_btn.setChecked(True)
            self._set_mask_tool("magic")

    def copy_layer(self):
        """Duplicate the active layer just above it (its undo history stays behind)."""
        i = self.active
        if i is None:
            return
        src = self.layers[i]
        src.settings = self._ui_settings()
        self._set_mask_tool(None)
        names = {L.name for L in self.layers}
        name, n = f"{src.name} copy", 2
        while name in names:
            name, n = f"{src.name} copy {n}", n + 1
        L = layers.Layer(name, copy.deepcopy(src.settings), src.work, src.full, visible=src.visible,
                         invert=src.invert, feather=src.feather, opacity=src.opacity, points=list(src.points),
                         ai=None if src.ai is None else src.ai.copy(), paint=src.paint.copy())
        self.layers.insert(i + 1, L)
        item = QListWidgetItem(L.name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if L.visible else Qt.CheckState.Unchecked)
        self.layer_list.blockSignals(True)
        self.layer_list.insertItem(i + 2, item)
        self.layer_list.blockSignals(False)
        self.set_active(i + 1)

    def delete_layer(self):
        i = self.active
        if i is None:
            return
        self._set_mask_tool(None)
        self.layer_list.setCurrentRow(0)  # back to the Base first
        del self.layers[i]
        self.layer_list.blockSignals(True)
        self.layer_list.takeItem(i + 1)
        self.layer_list.blockSignals(False)
        self.view.set_stack(self._stack())
        self.update_pipeline()  # a model only that layer used may stop
        self._layer_controls()

    def _clear_layers(self):
        """A new photo: layers belong to the old one."""
        if self.active is not None:
            self.set_active(None)
        self._set_mask_tool(None)
        self.layers.clear()
        self.layer_list.blockSignals(True)
        while self.layer_list.count() > 1:
            self.layer_list.takeItem(1)
        self.layer_list.setCurrentRow(0)
        self.layer_list.blockSignals(False)
        self.view.set_stack([])
        self.view.set_mask_show(None)
        self._layer_controls()

    def _layer_item_changed(self, item: QListWidgetItem):
        row = self.layer_list.row(item)
        if row >= 1:
            self.layers[row - 1].visible = item.checkState() == Qt.CheckState.Checked
            self._push_stack()

    def set_active(self, idx: int | None):
        """Which layer the tabs edit (None = Base)."""
        if idx == self.active:
            return
        ui = self._ui_settings()
        if self.active is None:
            self.base_settings = ui
        else:
            self.layers[self.active].settings = ui
        self.active = idx
        target = copy.deepcopy(self.base_settings if idx is None else self.layers[idx].settings)
        target["enhancement"]["red_eye"] = ui["enhancement"]["red_eye"]  # red eye is for the whole photo
        self._switching = True
        try:
            self.apply_settings(target, run_pipeline=False)
        finally:
            self._switching = False
        if idx is None:
            self.base_settings = None
        base = self._all_settings()[0]
        self.view.set_dev_params(dict(base["development"]))
        self.view.set_enh_params(self._enh_of(base))
        self.view.set_stack(self._stack())
        if self.layer_list.currentRow() != (0 if idx is None else idx + 1):
            self.layer_list.blockSignals(True)
            self.layer_list.setCurrentRow(0 if idx is None else idx + 1)
            self.layer_list.blockSignals(False)
        L = self.active_layer()
        if L is None:
            self._set_mask_tool(None)
        else:
            for wdg, val in ((self.invert_box, L.invert), (self.feather, round(L.feather)), (self.opacity, round(L.opacity * 100))):
                wdg.blockSignals(True)
                wdg.setChecked(val) if isinstance(wdg, QCheckBox) else wdg.setValue(val)
                wdg.blockSignals(False)
        self._show_mask()
        self._layer_controls()
        self.update_pipeline()

    def _layer_option(self, name: str, value):
        L = self.active_layer()
        if L is None:
            return
        setattr(L, name, float(value) if name != "invert" else bool(value))
        L.rebuild()
        self._mask_changed(L)

    def _show_mask(self):
        if self._hover is not None and any(L is self._hover for L in self.layers):  # hovered in the list
            self.view.set_mask_show(self._hover.mask)
            return
        L = self.active_layer()
        on = L is not None and (self.show_mask_box.isChecked() or self.view.mask_tool is not None)
        self.view.set_mask_show(L.mask if on else None)

    def _hover_item(self, item: QListWidgetItem | None):
        row = -1 if item is None else self.layer_list.row(item)
        L = self.layers[row - 1] if row >= 1 and row - 1 < len(self.layers) else None
        if L is not self._hover:
            self._hover = L
            self._show_mask()

    def eventFilter(self, obj, ev):
        if obj is self.layer_list.viewport():
            if ev.type() == QEvent.Type.Leave:
                self._hover_item(None)
            elif ev.type() == QEvent.Type.MouseMove and self.layer_list.itemAt(ev.position().toPoint()) is None:
                self._hover_item(None)
        return super().eventFilter(obj, ev)

    def _mask_changed(self, L: layers.Layer):
        if L is self.active_layer():
            self._show_mask()
        self.view.set_stack(self._stack())
        self._layer_controls()

    def _set_mask_tool(self, tool: str | None):
        if tool is not None and (self.active_layer() is None or not self._ensure_seg()):
            tool = None
        self.magic_btn.setChecked(tool == "magic")
        self.brush_btn.setChecked(tool == "brush")
        if tool is not None:
            self.crop_btn.setChecked(False)
            self.redeye_btn.setChecked(False)
        self.view.set_mask_tool(tool)
        self._show_mask()

    def _ensure_seg(self) -> bool:
        """Have the AI selection read this photo (once)."""
        if not segment.available():
            self.status.setText("AI selection model missing: run setup/download-models.sh")
            return False
        if self.seg_gen != self.photo_gen:
            self.seg_gen = self.photo_gen
            h, w = self.photo.linear.shape[:2]
            wh, ww = segment.work_shape(h, w)
            small = cv2.resize(self.photo.linear, (ww, wh), interpolation=cv2.INTER_AREA)
            self.view.set_mask_note("Reading the photo for AI selection…")
            self.start_seg.emit(self.photo_gen, small, self.photo.baseline_ev)
        return True

    def on_seg_ready(self, gen: int):
        if gen == self.photo_gen:
            self.view.set_mask_note("")

    def on_seg_failed(self, msg: str):
        self.seg_gen = -1
        self.view.set_mask_note("AI selection failed: " + msg)
        self.status.setText("AI selection failed")

    def _ask_mask(self, L: layers.Layer):
        if not L.points:
            return
        k = L.scale
        self.view.set_mask_note("Selecting…")
        self.start_mask.emit(self.photo_gen, L.uid, [(x * k, y * k) for x, y, _ in L.points],
                             [lab for _, _, lab in L.points])

    def on_mask_click(self, x: float, y: float, keep: bool):
        L = self.active_layer()
        if L is None:
            return
        L.snapshot()
        L.points.append((x, y, 1 if keep else 0))
        self._ask_mask(L)

    def on_mask_stroke(self, pts: list, radius: float, keep: bool):
        L = self.active_layer()
        if L is None or not pts:
            return
        L.snapshot()
        L.stroke(pts, radius, keep)
        # the stroke also guides the AI: its start, middle and end
        for i in sorted({0, len(pts) // 2, len(pts) - 1}):
            L.points.append((pts[i][0], pts[i][1], 1 if keep else 0))
        L.rebuild()
        self._mask_changed(L)  # the painted part shows at once
        self._ask_mask(L)

    def on_masked(self, gen: int, uid: int, prob: np.ndarray):
        if gen != self.photo_gen:
            return
        L = next((L for L in self.layers if L.uid == uid), None)
        if L is None:
            return
        L.ai = prob
        L.rebuild()
        self.view.set_mask_note("")
        self._mask_changed(L)

    def mask_undo(self):
        L = self.active_layer()
        if L is not None and L.undo_last():
            self._mask_changed(L)

    def mask_clear(self):
        L = self.active_layer()
        if L is not None:
            L.clear()
            self._mask_changed(L)

    # --- settings changes -----------------------------------------------
    def _reset(self, keys: list[str]):
        for key in keys:
            self.sliders[key][0].setValue(SLIDER_DEFAULTS[key])

    def reset_development(self):
        self._reset(RAW_KEYS + HSL_KEYS + HDR_KEYS + LOOK_KEYS)
        self.hdr_group.setChecked(bool(develop.DEFAULTS["hdr"]))
        self.refresh_luts(select=develop.DEFAULTS["lut"])

    def dev_changed(self, key: str, value: float):
        self.dev_params[key] = value
        if key.startswith("hsl_"):
            self._hsl_title()
        self._push_dev()

    def enh_changed(self, key: str, value: float):
        self.enh_params[key] = value
        self._push_enh()

    def _push_dev(self):
        """Development changed in the tabs: it belongs to the Base or the active layer."""
        if self._switching:
            return
        if self.active is None:
            self.view.set_dev_params(self.dev_params)
        else:
            self._push_stack()

    def _push_enh(self):
        if self._switching:
            return
        if self.active is None:
            self.view.set_enh_params(self.effective_enh())
        else:
            self._push_stack()

    def enh_settings_changed(self):
        self._sync_enh_view()
        if not self._applying:
            self.update_pipeline()

    def _sync_enh_view(self):
        self.sliders["ai_sharpen"][0].setEnabled(self.sharpen_box.currentData() is not None)
        self._push_enh()

    def effective_enh(self) -> dict:
        p = dict(self.enh_params)
        if not self.sh_group.isChecked():
            p["ai_sharpen"] = p["sharpen"] = 0.0
        return p

    def tab_changed(self, _):
        enhance = self.tabs.currentWidget() is self.enh_tab
        if not enhance:
            self.redeye_btn.setChecked(False)  # fixing by hand is on the Enhancement tab
        self.view.set_mode("enhance" if enhance else "develop")

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
            self.redeye_btn.setChecked(False)
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

    # --- red eye ----------------------------------------------------------
    def _set_redeye_controls(self):
        n = len(self.view.spots)
        ready = self.photo is not None
        self.redeye_btn.setEnabled(ready)
        self.redeye_undo.setEnabled(n > 0)
        self.redeye_again.setEnabled(ready and self.redeye_found is not None)
        if self.redeye_searching:
            text = "Looking for red eyes…"
        elif not ready or self.redeye_found is None:
            text = "Finds red eyes by itself."
        elif n:
            text = f"{n} red {'eye' if n == 1 else 'eyes'} fixed."
        else:
            text = "No red eyes found. To fix one anyway: Fix by hand."
        self.redeye_status.setText(text)

    def _redeye_search(self):
        if self.photo is None or self.redeye_searching or self.redeye_found is not None:
            return
        self.redeye_searching = True
        self.start_find.emit(self.photo_gen, self.photo.linear, self.photo.baseline_ev)
        self._set_redeye_controls()

    def redeye_section_toggled(self, on: bool):
        if on:
            self.view.set_spots(self.redeye_kept)
            self._redeye_search()
        else:
            self.redeye_btn.setChecked(False)
            self.redeye_kept = self.view.spots
            self.view.set_spots(())
        self._set_redeye_controls()

    def on_redeye_found(self, gen: int, spots: tuple):
        if gen != self.photo_gen:
            self.redeye_searching = False  # a search of the photo before this one
            if self.re_group.isChecked():
                self._redeye_search()
            return
        self.redeye_searching = False
        self.redeye_found = spots
        if self.re_group.isChecked():
            # fixes by hand made during the search stay, unless it found the same eye
            mine = [s for s in self.view.spots if not any(np.hypot(s.cx - f.cx, s.cy - f.cy) < f.radius for f in spots)]
            self.view.set_spots((*spots, *mine))
        else:
            self.redeye_kept = spots
        self._set_redeye_controls()

    def on_redeye_failed(self, gen: int, msg: str):
        self.redeye_searching = False
        if gen == self.photo_gen:
            self.redeye_found = ()
            self.status.setText(f"Red eye search failed: {msg}")
        self._set_redeye_controls()

    def redeye_find_again(self):
        self.view.set_spots(self.redeye_found or ())

    def redeye_toggled(self, on: bool):
        if on:
            self.crop_btn.setChecked(False)
        self.view.set_redeye_mode(on)
        self.view.setFocus()  # for Enter / Esc

    def on_redeye_click(self, x: float, y: float, r: float):
        lin, ev = self.photo.linear, self.photo.baseline_ev
        spot = redeye.find(lin, x, y, r, ev)
        if spot is None:
            self.view.set_redeye_note("No red pupil found there. Click closer to it, or scroll to change the search size.")
            return
        other = redeye.partner(lin, spot, ev)
        new = [s for s in (spot, other) if s is not None
               and not any(np.hypot(s.cx - o.cx, s.cy - o.cy) < o.radius for o in self.view.spots)]
        self.view.set_spots((*self.view.spots, *new))
        if not new:
            note = "Already fixed."
        elif other is None:
            note = "Fixed this eye; no red partner of the same size on either side."
        else:
            note = "Fixed both eyes."
        self.view.set_redeye_note(f"{note} Click the next eye, or press Enter when done.")

    # --- presets ----------------------------------------------------------
    def settings(self) -> dict:
        """The Base's settings: what presets, batches and the leave-photo question use."""
        if self.active is None or self.base_settings is None:
            return self._ui_settings()
        s = copy.deepcopy(self.base_settings)
        s["enhancement"]["red_eye"] = {"enabled": self.re_group.isChecked()}  # red eye is for the whole photo
        return s

    def _ui_settings(self) -> dict:
        """What the tabs show now (the Base's or the active layer's settings)."""
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
                "red_eye": {"enabled": self.re_group.isChecked()},
            },
        }

    def default_settings(self) -> dict:
        """Every setting at its default, in the same form as settings()."""
        def first(box: QComboBox):
            p = box.itemData(0) if box.count() else None
            return p.stem if p is not None else None

        d = SLIDER_DEFAULTS
        return {
            "development": dict(develop.DEFAULTS),
            "enhancement": {
                "noise_reduction": {"enabled": False, "model": first(self.denoise_box),
                                    "luminance": d["luminance"] / 100, "color": d["color"] / 100},
                "sharpening": {"enabled": False, "model": first(self.sharpen_box),
                               "ai_sharpen": d["ai_sharpen"] / 100, "sharpen": d["sharpen"] / 100},
                "red_eye": {"enabled": False},
            },
        }

    @staticmethod
    def _same_settings(a: dict, b: dict) -> bool:
        """Equal, ignoring float noise from slider steps."""
        def norm(x):
            if isinstance(x, dict):
                return {k: norm(v) for k, v in x.items() if k != "version"}
            if isinstance(x, float):
                return round(x, 4)
            return x
        return norm(a) == norm(b)

    def reset_all_settings(self, run_pipeline: bool = True):
        self.apply_settings(self.default_settings(), run_pipeline=run_pipeline)
        self.preset_box.setCurrentIndex(0)

    def apply_settings(self, s: dict, run_pipeline: bool = True):
        def set_slider(key, value):
            if value is not None:
                self.sliders[key][0].setValue(round(float(value) * 100))

        def set_model(box: QComboBox, stem):
            for i in range(box.count()):
                p = box.itemData(i)
                if (p.stem if p is not None else None) == stem:
                    box.setCurrentIndex(i)
                    return

        if self.active is not None and not self._switching:
            self.set_active(None)  # presets and resets are for the Base
        self._applying = True
        try:
            dev = s.get("development", {})
            for key in DEV_KEYS:
                set_slider(key, dev.get(key, develop.DEFAULTS[key]))
            self.hdr_group.setChecked(bool(dev.get("hdr", develop.DEFAULTS["hdr"])))
            self.bw_group.setChecked(bool(dev.get("bw", develop.DEFAULTS["bw"])))
            self.refresh_luts(select=str(dev.get("lut", develop.DEFAULTS["lut"]) or ""))
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
            self.re_group.setChecked(bool(enh.get("red_eye", {}).get("enabled", False)))
        finally:
            self._applying = False
        self._sync_enh_view()
        if run_pipeline:
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
            return False
        if name in presets.names() and name != current:
            if QMessageBox.question(self, "Save preset", f"Overwrite preset “{name}”?") != QMessageBox.StandardButton.Yes:
                return False
        presets.save(name, self.settings())
        self.refresh_presets(select=name)
        self.status.setText(f"Preset “{name}” saved")
        return True

    def delete_preset(self):
        name = self.preset_box.currentData()
        if not name:
            return
        if QMessageBox.question(self, "Delete preset", f"Delete preset “{name}”?") != QMessageBox.StandardButton.Yes:
            return
        presets.delete(name)
        self.refresh_presets()

    # --- edits kept with the photo (sidecar.py) -------------------------
    def edit_state(self) -> dict:
        """The open photo's whole edit, as written to its sidecar."""
        spots = self.view.spots if self.re_group.isChecked() else self.redeye_kept
        return {
            "photo": self.photo.path.name,
            "settings": self.settings(),
            "crop": [int(v) for v in self.view.crop] if self.view.crop else None,
            "layers": [{
                "name": L.name, "settings": st, "visible": L.visible, "invert": L.invert,
                "feather": float(L.feather), "opacity": float(L.opacity),
                "points": [[float(v) for v in p] for p in L.points],
                "ai": sidecar.pack(L.ai), "paint": sidecar.pack(L.paint),
            } for L, st in zip(self.layers, self._all_settings()[1:])],
            "red_eye_searched": self.redeye_found is not None,
            "red_eye": [{
                "x0": int(sp.x0), "y0": int(sp.y0), "cx": float(sp.cx), "cy": float(sp.cy),
                "radius": float(sp.radius), "mask": sidecar.pack(sp.mask), "dark": sidecar.pack(sp.dark),
            } for sp in spots],
        }

    def _has_edits(self) -> bool:
        return bool(self.view.crop or self.layers
                    or not self._same_settings(self.settings(), self.default_settings()))

    def keep_edits(self) -> bool:
        """Write the edit next to the photo (no edit: remove the file)."""
        if self.photo is None:
            return False
        try:
            if self._has_edits():
                p = sidecar.write(self.photo.path, self.edit_state())
                self.status.setText(f"Edits kept with the photo ({p.name})")
            else:
                sidecar.remove(self.photo.path)
                self.status.setText("No edits to keep: the photo is as shot")
            return True
        except OSError as e:
            QMessageBox.warning(self, "Could not keep edits",
                                f"Couldn't write {sidecar.path_for(self.photo.path)}:\n{e}")
            return False

    def _restore_edits(self, photo: imageio.Photo) -> bool:
        """Open the photo as it was left, if it has a sidecar."""
        try:
            data = sidecar.read(photo.path)
        except Exception as e:
            self.status.setText(f"Kept edits not read: {e}")
            return False
        if data is None:
            return False
        h, w = photo.linear.shape[:2]
        spots = tuple(redeye.Spot(int(d["x0"]), int(d["y0"]), sidecar.unpack(d["mask"]), sidecar.unpack(d["dark"]),
                                  float(d["cx"]), float(d["cy"]), float(d["radius"]))
                      for d in data.get("red_eye", []))
        if data.get("red_eye_searched"):
            self.redeye_found = spots  # no new search; Find again goes back to these
        self.redeye_kept = spots
        self.apply_settings(data["settings"], run_pipeline=False)
        if self.re_group.isChecked():
            self.view.set_spots(spots)
        if data.get("crop"):
            self.view.set_crop(tuple(data["crop"]))
        work = segment.work_shape(h, w)

        def fits(a):
            return a if a is not None and a.shape == tuple(work) else None

        for d in data.get("layers", []):
            st = copy.deepcopy(d["settings"])
            st["development"] = {**develop.DEFAULTS, **st.get("development", {})}
            paint = fits(sidecar.unpack(d.get("paint")))
            L = layers.Layer(d["name"], st, work, (h, w), visible=bool(d.get("visible", True)),
                             invert=bool(d.get("invert", False)), feather=float(d.get("feather", 0.0)),
                             opacity=float(d.get("opacity", 1.0)), points=[tuple(p) for p in d.get("points", [])],
                             ai=fits(sidecar.unpack(d.get("ai"))),
                             paint=None if paint is None else paint.astype(np.int8))
            self.layers.append(L)
            item = QListWidgetItem(L.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if L.visible else Qt.CheckState.Unchecked)
            self.layer_list.blockSignals(True)
            self.layer_list.addItem(item)
            self.layer_list.blockSignals(False)
        if self.layers:
            self.view.set_stack(self._stack())
        self._layer_controls()
        self.status.setText(f"Opened with the edits kept in {sidecar.path_for(photo.path).name}")
        return True

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

    def _settings_unsaved(self) -> bool:
        """Settings differ from the defaults and from the selected preset."""
        current = self.settings()
        if self._same_settings(current, self.default_settings()):
            return False
        name = self.preset_box.currentData()
        try:
            saved = presets.load(name) if name else None
            if saved is not None:  # presets from before red eye was a setting
                saved.setdefault("enhancement", {}).setdefault("red_eye", {"enabled": False})
            return not (saved and self._same_settings(current, saved))
        except Exception:
            return True

    def _leave_photo(self) -> bool:
        """Before another photo replaces the open one (or Freeroom quits): offer to
        keep the edits with the photo or as a preset, then reset everything.
        False = stay on this photo."""
        kept = sidecar.path_for(self.photo.path).exists()
        if kept:
            ask = not sidecar.same(self.photo.path, self.edit_state())
        else:  # settings that are a preset as they are, and nothing else, need no question
            ask = bool(self.view.crop or self.layers) or self._settings_unsaved()
        if ask:
            choice = self._ask_keep(kept)
            if choice == "keep":
                if not self.keep_edits():
                    return False
            elif choice == "preset":
                if not self.save_preset():
                    return False  # name dialog cancelled: keep everything as it was
            elif choice != "discard":
                return False  # Cancel or closed
        self.reset_all_settings(run_pipeline=False)  # the new photo starts the AI if needed
        return True

    def _ask_keep(self, kept: bool) -> str:
        """The leave-photo question: "keep", "preset", "discard" or "cancel"."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Keep edits?")
        box.setText(f"Keep the edits of {self.photo.path.name}?")
        box.setInformativeText(
            "Keep with photo: everything (settings, look, crop, layers) goes in a small file "
            f"next to it ({sidecar.path_for(self.photo.path).name}) and comes back when you "
            "open it again.\nSave as preset: only the settings, to use on other photos."
            + ("\n\nDiscard: the photo keeps the edits it had kept before." if kept else "")
        )
        keep_btn = box.addButton("Keep with photo", QMessageBox.ButtonRole.AcceptRole)
        preset_btn = box.addButton("Save as preset…", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton(QMessageBox.StandardButton.Discard)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(keep_btn)
        box.exec()
        return {keep_btn: "keep", preset_btn: "preset", discard_btn: "discard"}.get(box.clickedButton(), "cancel")

    def open_path(self, path: Path):
        """Start loading a photo (in the editor); on_loaded takes over when it's decoded."""
        if self.photo is not None and Path(path) != self.photo.path and not self._leave_photo():
            return
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
        self._clear_layers()
        self.photo = photo
        self.photo_gen += 1
        self.g = g
        self.library.show_folder(path.parent, select=path)
        self._show_shot(photo)
        self.enc_orig = None
        self.results.clear()
        self.running.clear()
        self.started.clear()
        self.save_target = None
        self.save_btn.setEnabled(True)
        self.keep_action.setEnabled(True)
        self.keep_btn.setEnabled(True)
        h, w, _ = photo.linear.shape
        kind = f"RAW, base exposure +{photo.baseline_ev:.2f} EV" if photo.meta.get("raw") else f"{photo.meta.get('bits')}-bit"
        self.setWindowTitle(f"Freeroom — {path.name} ({w}×{h}, {kind})")
        self.crop_btn.setEnabled(True)
        self._layer_controls()
        self.redeye_btn.setChecked(False)
        self.redeye_found, self.redeye_kept = None, ()
        self.view.set_linear(photo.linear, photo.baseline_ev, self.g)
        self._restore_edits(photo)
        if self.re_group.isChecked():
            self._redeye_search()
        self._set_redeye_controls()
        if not self.busy:
            self.update_pipeline()

    def _keys(self) -> tuple[tuple | None, tuple | None]:
        """Result keys the current settings need: (denoise, sharpen)."""
        dn_model = self.denoise_box.currentData() if self.nr_group.isChecked() else None
        sh_model = self.sharpen_box.currentData() if self.sh_group.isChecked() else None
        dn_key = (self.photo_gen, "dn", dn_model) if dn_model is not None else None
        sh_key = (self.photo_gen, "sh", sh_model, dn_model) if sh_model is not None else None
        return dn_key, sh_key

    @staticmethod
    def _model_path(box: QComboBox, stem) -> Path | None:
        for i in range(box.count()):
            p = box.itemData(i)
            if p is not None and p.stem == stem:
                return p
        return None

    def _keys_of(self, s: dict) -> tuple[tuple | None, tuple | None]:
        """Result keys stored settings need: (denoise, sharpen)."""
        nr, sh = s["enhancement"]["noise_reduction"], s["enhancement"]["sharpening"]
        dn_model = self._model_path(self.denoise_box, nr.get("model")) if nr.get("enabled") else None
        sh_model = self._model_path(self.sharpen_box, sh.get("model")) if sh.get("enabled") else None
        dn_key = (self.photo_gen, "dn", dn_model) if dn_model is not None else None
        sh_key = (self.photo_gen, "sh", sh_model, dn_model) if sh_model is not None else None
        return dn_key, sh_key

    @staticmethod
    def _enh_of(s: dict) -> dict:
        """Enhancement blend parameters of stored settings (like effective_enh)."""
        nr, sh = s["enhancement"]["noise_reduction"], s["enhancement"]["sharpening"]
        on = bool(sh.get("enabled"))
        return {"luminance": nr["luminance"], "color": nr["color"],
                "ai_sharpen": sh["ai_sharpen"] if on else 0.0, "sharpen": sh["sharpen"] if on else 0.0}

    def _all_settings(self) -> list[dict]:
        """Settings of the Base, then of every layer (the tabs' for the one edited)."""
        ui = self._ui_settings()
        out = [ui if self.active is None or self.base_settings is None else self.base_settings]
        out += [ui if i == self.active else L.settings for i, L in enumerate(self.layers)]
        return out

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
        dn_key, sh_key = self._keys_of(self._all_settings()[0])
        self.view.set_layers(self._layer(dn_key), self._layer(sh_key))
        if self.layers:
            self._push_stack()

    def stop_enhancement(self):
        """Cancel button: untick the sections whose AI step isn't done (sharpening
        first, as unticking noise reduction would make it start over), which stops
        them; a save waiting for them is dropped."""
        self.save_target = None
        for group, which in ((self.sh_group, 1), (self.nr_group, 0)):
            key = self._keys()[which]
            if key is not None and key not in self.results:
                group.setChecked(False)

    def update_pipeline(self):
        """Run the AI steps the ticked sections of the Base and every layer need
        that haven't run yet for this photo, then save if a save is waiting."""
        if self.photo is None or self.denoise_box.count() == 0:
            return
        pairs = [self._keys_of(st) for st in self._all_settings()]
        wanted = frozenset(k for pair in pairs for k in pair if k is not None)
        if self.busy:
            # Steps that were switched off (or got another model) stop after their
            # current tile; the rest of the job goes on and its results are kept.
            self.worker.wanted = wanted
        self._show_layers()  # the Raw tab shows the results there are, too
        if self.busy:
            self.pending = True  # come back here when the job ends
            return
        stages, seen = [], set()
        if any(k not in self.results for k in wanted) and self.enc_orig is None:
            self.enc_orig = pipeline.encode(self.photo.linear, self.g)
        for dn_key, _ in pairs:
            if dn_key is not None and dn_key not in self.results and dn_key not in seen:
                seen.add(dn_key)
                dst = self.enc_orig.copy()  # shows the unprocessed photo until tiles land
                self.running[dn_key] = dst
                stages.append((dn_key, "Denoising", dn_key[2], self.enc_orig, dst))
        for dn_key, sh_key in pairs:
            if sh_key is not None and sh_key not in self.results and sh_key not in seen:
                seen.add(sh_key)
                src = self.enc_orig if dn_key is None else self.results.get(dn_key, self.running.get(dn_key))
                dst = np.empty_like(src)  # filled by the worker once that stage starts
                self.running[sh_key] = dst
                stages.append((sh_key, "Sharpening", sh_key[2], src, dst))

        if not stages:
            if self.save_target is not None:
                self._do_save()
            return
        self.worker.cancel = False
        self.worker.wanted = wanted
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
        """Stop the whole run (a batch needs the GPU); sections stay ticked."""
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
        if ok and self.worker.ran:
            self.status.setText(f"Done in {seconds:.1f} s · {engine.device_name()}")
        elif ok:
            self.status.setText("Stopped: the AI step was switched off")
        asked_again, self.pending = self.pending, False
        self._show_layers()
        if self.cancelled and not ok and not asked_again:
            self.status.setText("Cancelled. Untick and tick a section to run it again.")
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
        save_now = False
        if self.busy:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle("Enhancement running")
            box.setText("An enhancement is still running.")
            box.setInformativeText(
                "Save now: the photo is saved with the AI steps that have finished; the running "
                "one goes on.\nWait: the photo is saved when the enhancement is done."
            )
            now_btn = box.addButton("Save now", QMessageBox.ButtonRole.AcceptRole)
            wait_btn = box.addButton("Wait, then save", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(wait_btn)
            box.exec()
            if box.clickedButton() not in (now_btn, wait_btn):
                return
            save_now = box.clickedButton() is now_btn
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        suggested = SAVE_DIR / f"{self.photo.path.stem}_edit.jpg"
        filters = {f"{name} (*{ext})": (ext, opts) for name, (ext, opts) in batch.FORMATS.items()}
        path, chosen = QFileDialog.getSaveFileName(self, "Save photo", str(suggested), ";;".join(filters))
        if not path:
            return
        out = Path(path)
        ext, opts = filters.get(chosen, (None, {}))
        # the picked format's options, unless the name typed asks for another format
        same = ext and out.suffix.lower() in (ext, ext.replace(".tif", ".tiff"), ext.replace(".jpg", ".jpeg"))
        self.save_options = dict(opts) if same else {}
        if out.suffix.lower() not in imageio.SAVE_EXTS:
            out = out.with_suffix(".jpg")
        if out.resolve() == self.photo.path.resolve():
            QMessageBox.warning(self, "Could not save", "Refusing to overwrite the original photo.")
            return
        if save_now:  # with the AI steps that are done; the running one goes on
            self.save_target = out
            self._do_save()
            return
        # Runs whatever enhancement is still missing, then saves (see update_pipeline).
        self.save_target = out
        if self.busy:
            self.status.setText(f"Saving {out.name} when the enhancement is done…")
        else:
            self.status.setText("Preparing full-size image for saving…")
        self.update_pipeline()

    def _do_save(self):
        """Hand the full-size develop + write to the saver thread, with the
        settings as they are now (changing sliders meanwhile doesn't affect it)."""
        out, self.save_target = self.save_target, None
        base = self._all_settings()[0]
        dn_key, sh_key = self._keys_of(base)
        options = self.save_options
        self.saving = True
        self.save_btn.setEnabled(False)
        self.view.set_progress(f"Saving {out.name}…", 0)
        self.start_save.emit({
            "photo": self.photo, "out": out, "g": self.g, "options": options,
            "dn": self.results.get(dn_key), "sharp": self.results.get(sh_key),
            "dev": dict(base["development"]), "enh": self._enh_of(base), "crop": self.view.crop,
            "layers": self._stack(finished_only=True),
            "spots": None if self.re_group.isChecked() and self.redeye_found is None else self.view.spots,
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
        if self.photo is not None and not self._leave_photo():
            e.ignore()
            return
        self.worker.cancel = True
        self.thread.quit()
        self.thread.wait(5000)
        self.load_thread.quit()
        self.load_thread.wait(5000)
        self.seg_thread.quit()
        self.seg_thread.wait(5000)
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
