"""Library view: browse folders, see thumbnails and a large preview of photos
(RAW included), open one in the editor.

Photo folders are only ever read. RAW thumbnails/previews come from the JPEG
the camera embedded in the file (fast, no demosaicing); thumbnails are cached
in the system cache folder (~/.cache/freeroom/thumbs on Linux,
~/Library/Caches/freeroom/thumbs on macOS).
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QDir,
    QFileInfo,
    QModelIndex,
    QObject,
    QRunnable,
    QSettings,
    QSize,
    QStandardPaths,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import QColor, QIcon, QImage, QKeySequence, QPainter, QPixmap, QShortcut, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QFileIconProvider,
    QFileSystemModel,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from . import develop, imageio

THUMB_SIZE = 320  # pixels stored per thumbnail (the grid scales them down)
PREVIEW_SIZE = 1600
# ~/.cache on Linux, ~/Library/Caches on macOS
CACHE_DIR = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericCacheLocation)) / "freeroom" / "thumbs"
PATH_ROLE = Qt.ItemDataRole.UserRole + 1

# LibRaw "flip" -> rotation of the embedded preview (it's stored unrotated)
_FLIP = {3: cv2.ROTATE_180, 5: cv2.ROTATE_90_COUNTERCLOCKWISE, 6: cv2.ROTATE_90_CLOCKWISE}


# --- reading previews ------------------------------------------------------
def _fit(im: Image.Image, size: int) -> Image.Image:
    im = im.convert("RGB")
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    return im


def _raw_preview(path: Path, size: int) -> tuple[Image.Image, dict]:
    import rawpy

    info: dict = {}
    try:
        with rawpy.imread(str(path)) as raw:
            s = raw.sizes
            flip = s.flip
            info["pixels"] = (s.height, s.width) if flip in (5, 6) else (s.width, s.height)
            try:
                thumb = raw.extract_thumb()
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                thumb = None
            if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
                im = Image.open(io.BytesIO(thumb.data))
                im.draft("RGB", (size, size))  # JPEG: decode at reduced size, much faster
                arr = np.asarray(im.convert("RGB"))
            elif thumb is not None and thumb.format == rawpy.ThumbFormat.BITMAP:
                arr = thumb.data
            else:
                # No usable embedded preview: a quick half-size development (already rotated)
                return _fit(Image.fromarray(raw.postprocess(half_size=True, use_camera_wb=True)), size), info
    except rawpy.LibRawFileUnsupportedError:
        # e.g. JPEG XL DNGs: full load via Freeroom's own reader, default development
        photo = imageio.load(path)
        lin = photo.linear[:: max(1, photo.linear.shape[0] // size)][:, :: max(1, photo.linear.shape[1] // size)]
        rgb = develop.develop(np.ascontiguousarray(lin), develop.DEFAULTS, photo.baseline_ev)
        info["pixels"] = photo.linear.shape[1::-1]
        return _fit(Image.fromarray((rgb * 255 + 0.5).astype(np.uint8)), size), info
    if flip in _FLIP:
        arr = cv2.rotate(np.ascontiguousarray(arr), _FLIP[flip])
    return _fit(Image.fromarray(arr), size), info


def _raster_preview(path: Path, size: int) -> tuple[Image.Image, dict]:
    info: dict = {}
    if path.suffix.lower() in {".jpg", ".jpeg", ".webp", ".bmp"}:
        with Image.open(path) as im:
            info["pixels"] = im.size
            exif = im.getexif()
            if exif:
                sub = exif.get_ifd(0x8769)
                info["taken"] = sub.get(36867) or exif.get(306)
                make, model = (str(exif.get(t, "")).strip() for t in (271, 272))
                info["camera"] = model if model.lower().startswith(make.lower()) else f"{make} {model}".strip()
                if exif.get(imageio.EXIF_ORIENTATION) in (5, 6, 7, 8):
                    info["pixels"] = im.size[::-1]
            im.draft("RGB", (size, size))
            return _fit(ImageOps.exif_transpose(im), size), info
    # PNG / TIFF (possibly 16-bit): OpenCV converts to 8-bit and applies orientation
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("unreadable image")
    info["pixels"] = (bgr.shape[1], bgr.shape[0])
    return _fit(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), size), info


def read_preview(path: Path, size: int) -> tuple[Image.Image, dict]:
    if path.suffix.lower() in imageio.RAW_EXTS:
        return _raw_preview(path, size)
    return _raster_preview(path, size)


def _cache_file(path: Path, st: os.stat_result) -> Path:
    key = hashlib.sha1(f"{path}|{st.st_mtime_ns}|{st.st_size}|{THUMB_SIZE}".encode()).hexdigest()
    return CACHE_DIR / key[:2] / f"{key}.jpg"


def thumbnail(path: Path) -> Image.Image:
    st = path.stat()
    cached = _cache_file(path, st)
    try:
        with Image.open(cached) as im:
            return im.convert("RGB")
    except OSError:
        pass
    im, _ = read_preview(path, THUMB_SIZE)
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".part")
        im.save(tmp, "JPEG", quality=85)
        tmp.replace(cached)
    except OSError:
        pass  # cache is only a speed-up
    return im


def to_qimage(im: Image.Image) -> QImage:
    data = im.tobytes()
    return QImage(data, im.width, im.height, 3 * im.width, QImage.Format.Format_RGB888).copy()


# --- background jobs -------------------------------------------------------
class _Signals(QObject):
    thumb = Signal(int, str, QImage)  # generation, path, image (null = failed)
    preview = Signal(int, str, QImage, dict, str)  # request, path, image, info, error


class _ThumbJob(QRunnable):
    def __init__(self, sig: _Signals, gen: int, current, path: Path):
        super().__init__()
        self.sig, self.gen, self.current, self.path = sig, gen, current, path

    def run(self):
        if self.current() != self.gen:
            return  # another folder was opened meanwhile
        try:
            img = to_qimage(thumbnail(self.path))
        except Exception:
            img = QImage()
        self.sig.thumb.emit(self.gen, str(self.path), img)


class _PreviewJob(QRunnable):
    def __init__(self, sig: _Signals, req: int, current, path: Path):
        super().__init__()
        self.sig, self.req, self.current, self.path = sig, req, current, path

    def run(self):
        if self.current() != self.req:
            return
        try:
            im, info = read_preview(self.path, PREVIEW_SIZE)
            info["shot"] = imageio.shot_info(self.path)
            self.sig.preview.emit(self.req, str(self.path), to_qimage(im), info, "")
        except Exception as e:
            self.sig.preview.emit(self.req, str(self.path), QImage(), {}, f"{type(e).__name__}: {e}")


# --- widgets ---------------------------------------------------------------
class PreviewLabel(QLabel):
    """Shows a QImage scaled to fit, keeping its aspect ratio."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 200)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #1e1e20; color: #888;")
        self.img = QImage()

    def set_image(self, img: QImage, text: str = ""):
        self.img = img
        self.setText(text if img.isNull() else "")
        self._rescale()

    def _rescale(self):
        if self.img.isNull():
            self.setPixmap(QPixmap())
            return
        dpr = self.devicePixelRatioF()
        pm = QPixmap.fromImage(self.img).scaled(
            self.size() * dpr, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        pm.setDevicePixelRatio(dpr)
        self.setPixmap(pm)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()


def _placeholder(size: int, text: str = "") -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(40, 40, 44))
    if text:
        p = QPainter(pm)
        p.setPen(QColor(130, 130, 130))
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, text)
        p.end()
    return QIcon(pm)


def _places() -> list[tuple[str, Path]]:
    """Pictures, Home and mounted drives (Linux: /run/media/$USER, /media/$USER; macOS: /Volumes)."""
    home = Path.home()
    out = [("Pictures", home / "Pictures"), ("Home", home)]
    if sys.platform == "darwin":
        roots = [Path("/Volumes")]
    else:
        roots = [Path("/run/media") / home.name, Path("/media") / home.name]
    for root in roots:
        if root.is_dir():
            for d in sorted(root.iterdir()):
                # skip the Mac's own startup disk (/Volumes/Macintosh HD -> /)
                if d.is_dir() and not (d.is_symlink() and d.resolve() == Path("/")):
                    out.append((d.name, d))
    return [p for p in out if p[1].is_dir()]


class LibraryView(QWidget):
    """Folder tree | thumbnail grid | large preview."""

    open_photo = Signal(object)  # Path

    def __init__(self):
        super().__init__()
        self.settings = QSettings("freeroom", "freeroom")
        self.folder: Path | None = None
        self.gen = 0  # folder generation: stale thumbnails are dropped
        self.preview_req = 0
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(4)
        self.sig = _Signals()
        self.sig.thumb.connect(self.on_thumb)
        self.sig.preview.connect(self.on_preview)
        self.items: dict[str, QStandardItem] = {}
        self._select_after_scan: Path | None = None

        # --- left: places + folder tree
        self.places = QListWidget()
        self.places.setMaximumHeight(130)
        icons = QFileIconProvider()  # the platform's own folder / drive icons
        for name, path in _places():
            it = QListWidgetItem(icons.icon(QFileInfo(str(path))), name)
            it.setData(PATH_ROLE, str(path))
            it.setToolTip(str(path))
            self.places.addItem(it)
        self.places.itemClicked.connect(lambda it: self.show_folder(Path(it.data(PATH_ROLE))))

        self.fs = QFileSystemModel(self)
        self.fs.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot)
        self.fs.setRootPath("/")
        self.tree = QTreeView()
        self.tree.setModel(self.fs)
        for col in (1, 2, 3):
            self.tree.hideColumn(col)
        self.tree.setHeaderHidden(True)
        self.tree.clicked.connect(lambda idx: self.show_folder(Path(self.fs.filePath(idx)), sync_tree=False))
        left = QWidget()
        lay = QVBoxLayout(left)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("<b>Places</b>"))
        lay.addWidget(self.places)
        lay.addWidget(QLabel("<b>Folders</b>"))
        lay.addWidget(self.tree, 1)

        # --- middle: thumbnail grid
        self.path_label = QLabel()
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.count_label = QLabel()
        self.count_label.setStyleSheet("color: #999;")
        self.raw_only = QCheckBox("RAW only")
        self.raw_only.setToolTip("Show only RAW files (" + ", ".join(sorted(e.lstrip(".").upper() for e in imageio.RAW_EXTS)) + ")")
        self.raw_only.setChecked(self.settings.value("library/raw_only", False, type=bool))
        self.raw_only.toggled.connect(self._raw_only_toggled)
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(100, 320)
        self.size_slider.setValue(int(self.settings.value("library/thumb", 170)))
        self.size_slider.setFixedWidth(120)
        self.size_slider.setToolTip("Thumbnail size")
        self.size_slider.valueChanged.connect(self._apply_thumb_size)

        self.model = QStandardItemModel(self)
        self.grid = QListView()
        self.grid.setModel(self.model)
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setMovement(QListView.Movement.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setWordWrap(False)
        self.grid.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.grid.setSpacing(6)
        self.grid.setStyleSheet("QListView { background: #1e1e20; color: #ccc; }")
        self.grid.selectionModel().currentChanged.connect(self.on_current)
        self.grid.doubleClicked.connect(self._open_index)
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            sc = QShortcut(QKeySequence(key), self.grid)
            sc.setContext(Qt.ShortcutContext.WidgetShortcut)
            sc.activated.connect(lambda: self._open_index(self.grid.currentIndex()))

        bar = QHBoxLayout()
        bar.addWidget(self.path_label, 1)
        bar.addWidget(self.count_label)
        bar.addWidget(self.raw_only)
        bar.addWidget(QLabel("Size"))
        bar.addWidget(self.size_slider)
        middle = QWidget()
        lay = QVBoxLayout(middle)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.grid, 1)

        # --- right: preview + info
        self.preview = PreviewLabel()
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.open_btn = QPushButton("Open in editor")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(lambda: self._open_index(self.grid.currentIndex()))
        right = QWidget()
        lay = QVBoxLayout(right)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.preview, 1)
        lay.addWidget(self.info)
        lay.addWidget(self.open_btn)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(middle)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 2)
        split.setSizes([260, 760, 480])
        root = QHBoxLayout(self)
        root.addWidget(split)

        self._apply_thumb_size(self.size_slider.value())
        start = Path(str(self.settings.value("library/folder", Path.home() / "Pictures")))
        self.show_folder(start if start.is_dir() else Path.home())

    # --- folder ---------------------------------------------------------
    def show_folder(self, folder: Path, select: Path | None = None, sync_tree: bool = True):
        folder = Path(folder)
        if not folder.is_dir():
            return
        if folder != self.folder:
            self.folder = folder
            self.settings.setValue("library/folder", str(folder))
            self._select_after_scan = select
            self.refresh()
        elif select is not None:
            self.select(select)
        if sync_tree:
            idx = self.fs.index(str(folder))
            self.tree.setCurrentIndex(idx)
            self.tree.scrollTo(idx)
            self.tree.expand(idx)

    def refresh(self):
        """(Re)list the current folder and start loading its thumbnails."""
        self.gen += 1
        self.pool.clear()  # queued jobs of the old folder
        self.model.clear()
        self.items.clear()
        folder = self.folder
        self.path_label.setText(f"<b>{folder.name or folder}</b> <span style='color:#999'>{folder}</span>")
        exts = imageio.RAW_EXTS if self.raw_only.isChecked() else imageio.ALL_EXTS
        try:
            files = sorted(
                (Path(e.path) for e in os.scandir(folder) if e.is_file() and Path(e.name).suffix.lower() in exts),
                key=lambda p: p.name.lower(),
            )
        except OSError as e:
            files = []
            self.count_label.setText(f"Can't read folder: {e.strerror}")
        else:
            self.count_label.setText(f"{len(files)} photo{'s' if len(files) != 1 else ''}")
        icon = _placeholder(THUMB_SIZE)
        gen = self.gen
        for path in files:
            it = QStandardItem(icon, path.name)
            it.setEditable(False)
            it.setData(str(path), PATH_ROLE)
            it.setToolTip(path.name)
            self.model.appendRow(it)
            self.items[str(path)] = it
            self.pool.start(_ThumbJob(self.sig, gen, lambda: self.gen, path))
        target = self._select_after_scan
        self._select_after_scan = None
        if target is not None and str(target) in self.items:
            self.select(target)
        elif files:
            self.grid.setCurrentIndex(self.model.index(0, 0))
        else:
            self.preview.set_image(QImage(), "No photos in this folder")
            self.info.setText("")
            self.open_btn.setEnabled(False)

    def _raw_only_toggled(self, on: bool):
        self.settings.setValue("library/raw_only", on)
        cur = self.grid.currentIndex()
        self._select_after_scan = Path(cur.data(PATH_ROLE)) if cur.isValid() else None  # keep it if still shown
        self.refresh()

    def select(self, path: Path):
        it = self.items.get(str(path))
        if it is not None:
            idx = it.index()
            self.grid.setCurrentIndex(idx)
            self.grid.scrollTo(idx)

    def on_thumb(self, gen: int, path: str, img: QImage):
        if gen != self.gen or path not in self.items:
            return
        self.items[path].setIcon(QIcon(QPixmap.fromImage(img)) if not img.isNull() else _placeholder(THUMB_SIZE, "?"))

    def _apply_thumb_size(self, size: int):
        self.settings.setValue("library/thumb", size)
        self.grid.setIconSize(QSize(size, size))
        self.grid.setGridSize(QSize(size + 16, size + 34))

    # --- preview --------------------------------------------------------
    def on_current(self, cur: QModelIndex, _prev=None):
        if not cur.isValid():
            return
        path = Path(cur.data(PATH_ROLE))
        self.open_btn.setEnabled(True)
        self.preview_req += 1
        icon = self.items[str(path)].icon()
        self.preview.set_image(icon.pixmap(THUMB_SIZE).toImage())  # thumbnail until the big one is ready
        self.info.setText(self._info_text(path, {}))
        req = self.preview_req
        self.pool.start(_PreviewJob(self.sig, req, lambda: self.preview_req, path), 1)  # before thumbnails

    def on_preview(self, req: int, path: str, img: QImage, info: dict, error: str):
        if req != self.preview_req:
            return
        if error:
            self.preview.set_image(QImage(), f"No preview\n{error}")
        else:
            self.preview.set_image(img)
        self.info.setText(self._info_text(Path(path), info))

    @staticmethod
    def _info_text(path: Path, info: dict) -> str:
        try:
            st = path.stat()
        except OSError:
            return f"<b>{path.name}</b>"
        kind = "RAW" if path.suffix.lower() in imageio.RAW_EXTS else path.suffix.lstrip(".").upper()
        lines = [f"<b>{path.name}</b>", f"{kind} · {st.st_size / 1e6:.1f} MB"]
        if "pixels" in info:
            w, h = info["pixels"]
            lines[1] += f" · {w}×{h} ({w * h / 1e6:.1f} MP)"
        shot = info.get("shot", {})
        head, _ = imageio.describe_shot(shot)
        if head:
            lines.append(head)
        if camera := shot.get("camera") or info.get("camera"):
            lines.append(f"{camera} · {shot['lens']}" if shot.get("lens") else str(camera))
        lines.append(shot["taken"].strftime("Taken %Y-%m-%d %H:%M") if shot.get("taken") else
                     time.strftime("Modified %Y-%m-%d %H:%M", time.localtime(st.st_mtime)))
        return "<br>".join(lines)

    def _open_index(self, idx: QModelIndex):
        if idx.isValid():
            self.open_photo.emit(Path(idx.data(PATH_ROLE)))

    def shutdown(self):
        self.gen += 1
        self.preview_req += 1
        self.pool.clear()
        self.pool.waitForDone(3000)
