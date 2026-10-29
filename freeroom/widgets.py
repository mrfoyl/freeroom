"""Photo view with the 100% loupe and crop tool, the RGB range view (histogram), and sliders."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QLabel, QProgressBar, QSlider, QVBoxLayout, QWidget

from . import develop, pipeline


def to_qimage(rgb: np.ndarray, dpr: float = 1.0) -> QImage:
    arr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    h, w, _ = arr.shape
    img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    img.setDevicePixelRatio(dpr)
    return img


class ResetSlider(QSlider):
    """Horizontal slider that jumps back to its default on double-click."""

    def __init__(self, default: int):
        super().__init__(Qt.Orientation.Horizontal)
        self.default = default
        self.setToolTip("Double-click to reset")

    def mouseDoubleClickEvent(self, e):
        self.setValue(self.default)
        e.accept()


class _Renderer(QObject):
    """Renders the screen-size view off the GUI thread, so sliders stay smooth."""

    done = Signal(object, object, object, float)  # QImage, rgb, histogram, beyond-white fraction

    def render(self, job):
        render, dpr, bw_params = job
        out = render()
        bw = develop.beyond_white(*bw_params) if bw_params else 0.0
        self.done.emit(to_qimage(out, dpr), out, develop.histogram(out[::2, ::2]), bw)


class PhotoView(QWidget):
    """Fit-to-window preview with a circular 100% loupe under the mouse.

    Everything is rendered on the fly from the linear photo (+ the AI layers,
    which are stored in the pipeline encoding), so dials are always instant.
    Two modes, one per tab:
      "develop": developed photo (before = default development)
      "enhance": developed photo + AI layers (before = developed photo)

    Crop: `crop` = (x0, y0, x1, y1) in photo pixels, or None. In crop mode the
    whole photo is shown with the crop frame to edit; otherwise only the
    cropped part is shown (and the loupe stays inside it).
    """

    # screen-size result, its histogram, fraction of RAW data past white (develop mode)
    rendered = Signal(object, object, float)
    crop_changed = Signal(object)  # (x0, y0, x1, y1) or None
    crop_finished = Signal(bool)  # True = keep (Enter), False = cancel (Esc)
    _request = Signal(object)

    HANDLE = 9  # px: grab distance of the crop handles
    MIN_CROP = 16  # photo px

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 300)
        self.mode = "develop"
        self.linear: np.ndarray | None = None
        self.baseline_ev = 0.0
        self.g = 1.0  # pipeline encoding scale of the AI layers
        self.dn: np.ndarray | None = None  # denoised, encoded
        self.sharp: np.ndarray | None = None  # sharpened, encoded
        self.dev_params = dict(develop.DEFAULTS)
        self.enh_params: dict = {}
        self._lut = self._default_lut = develop.build_lut(develop.DEFAULTS)
        self._hdr_map: np.ndarray | None = None  # of the full photo, made when HDR is first on
        # view state
        self.radius = 160
        self.mouse: QPointF | None = None
        self.show_orig_loupe = False  # mouse button held
        self.show_orig_all = False  # space held
        self._small: dict[str, np.ndarray | None] = {}
        self._view_img: QImage | None = None
        self._view_orig: QImage | None = None
        self._rect = QRectF()
        # crop
        self.crop: tuple[int, int, int, int] | None = None
        self.crop_mode = False
        self.crop_aspect: float | None = None  # width / height, None = free
        self._drag: tuple | None = None  # (kind, start photo point, start crop)
        # Loading overlay in the middle of the view (busy bar: decoders don't report progress)
        self._loading = QWidget(self)
        self._loading_text = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._loading_text.setStyleSheet("color: #ddd; font-size: 12pt; background: transparent;")
        self._loading_bar = bar = QProgressBar()
        bar.setRange(0, 0)
        bar.setTextVisible(False)
        bar.setFixedSize(320, 14)
        lay = QVBoxLayout(self._loading)
        lay.addWidget(self._loading_text)
        lay.addWidget(bar, alignment=Qt.AlignmentFlag.AlignCenter)
        self._loading.setStyleSheet("QWidget { background: rgba(20, 20, 22, 210); border-radius: 8px; }")
        self._loading.hide()
        # Background rendering: at most one job at a time; changes meanwhile
        # mark it dirty and the newest state is rendered when the job ends.
        self._rendering = False
        self._dirty = False
        self._render_thread = QThread(self)
        self._renderer = _Renderer()
        self._renderer.moveToThread(self._render_thread)
        self._request.connect(self._renderer.render)
        self._renderer.done.connect(self._render_done)
        self._render_thread.start()

    def shutdown(self):
        self._render_thread.quit()
        self._render_thread.wait(5000)

    def set_loading(self, text: str | None):
        """Show (text) or hide (None) the overlay with a busy bar."""
        self.set_progress(text, None)

    def set_progress(self, text: str | None, percent: int | None = None):
        """Show (text) or hide (None) the overlay; percent None = busy bar."""
        if text is None:
            self._loading.hide()
        else:
            if percent is None:
                self._loading_bar.setRange(0, 0)
            else:
                self._loading_bar.setRange(0, 100)
                self._loading_bar.setValue(percent)
            self._loading_text.setText(text)
            self._loading.adjustSize()
            self._place_loading()
            self._loading.show()
        self.update()

    def is_loading(self) -> bool:
        return self._loading.isVisible()

    def _place_loading(self):
        w = self._loading
        w.move((self.width() - w.width()) // 2, (self.height() - w.height()) // 2)

    # --- data -----------------------------------------------------------
    def set_linear(self, linear: np.ndarray, baseline_ev: float, g: float):
        self.linear, self.baseline_ev, self.g = linear, baseline_ev, g
        self.dn = self.sharp = None
        self._hdr_map = None
        self.crop = None
        self.crop_changed.emit(None)
        self._small = {}
        self._rebuild_small()

    # --- crop -------------------------------------------------------------
    def set_crop_mode(self, on: bool):
        self.crop_mode = on
        self._drag = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._small = {}  # the shown region changes
        self._rebuild_small()

    def set_crop(self, crop: tuple | None):
        if crop is not None:
            h, w, _ = self.linear.shape
            x0, y0, x1, y1 = (int(round(v)) for v in crop)
            x0, x1 = sorted((min(max(x0, 0), w), min(max(x1, 0), w)))
            y0, y1 = sorted((min(max(y0, 0), h), min(max(y1, 0), h)))
            crop = None if (x0, y0, x1, y1) == (0, 0, w, h) else (x0, y0, x1, y1)
        self.crop = crop
        self.crop_changed.emit(crop)
        if self.crop_mode:
            self.update()
        else:
            self._small = {}
            self._rebuild_small()

    def set_crop_aspect(self, aspect: float | None):
        """Constrain the crop to width/height = aspect (None = free); the current
        crop keeps its centre and roughly its area."""
        self.crop_aspect = aspect
        if aspect is None or self.linear is None:
            return
        h, w, _ = self.linear.shape
        x0, y0, x1, y1 = self.crop or (0, 0, w, h)
        cx, cy, area = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * (y1 - y0)
        cw = np.sqrt(area * aspect)
        ch = cw / aspect
        k = min(1.0, w / cw, h / ch)  # shrink to fit the photo
        cw, ch = cw * k, ch * k
        cx = min(max(cx, cw / 2), w - cw / 2)
        cy = min(max(cy, ch / 2), h - ch / 2)
        self.set_crop((cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2))

    def _region(self) -> tuple[int, int, int, int]:
        """Part of the photo shown: everything in crop mode, else the crop."""
        h, w, _ = self.linear.shape
        if self.crop_mode or self.crop is None:
            return (0, 0, w, h)
        return self.crop

    def _to_photo(self, pt: QPointF) -> tuple[float, float]:
        x0, y0, x1, y1 = self._region()
        return (x0 + (pt.x() - self._rect.x()) / self._rect.width() * (x1 - x0),
                y0 + (pt.y() - self._rect.y()) / self._rect.height() * (y1 - y0))

    def _to_widget(self, x: float, y: float) -> QPointF:
        x0, y0, x1, y1 = self._region()
        return QPointF(self._rect.x() + (x - x0) / (x1 - x0) * self._rect.width(),
                       self._rect.y() + (y - y0) / (y1 - y0) * self._rect.height())

    def _crop_box(self) -> tuple[int, int, int, int]:
        h, w, _ = self.linear.shape
        return self.crop or (0, 0, w, h)

    def _hit(self, pt: QPointF) -> str | None:
        """Which part of the crop frame is under pt: corner/edge (n, s, e, w,
        ne, …), 'move' inside, None outside."""
        x0, y0, x1, y1 = self._crop_box()
        a, b = self._to_widget(x0, y0), self._to_widget(x1, y1)
        near = lambda u, v: abs(u - v) <= self.HANDLE  # noqa: E731
        inside_x = a.x() - self.HANDLE <= pt.x() <= b.x() + self.HANDLE
        inside_y = a.y() - self.HANDLE <= pt.y() <= b.y() + self.HANDLE
        if not (inside_x and inside_y):
            return None
        v = "n" if near(pt.y(), a.y()) else "s" if near(pt.y(), b.y()) else ""
        h = "w" if near(pt.x(), a.x()) else "e" if near(pt.x(), b.x()) else ""
        return (v + h) or ("move" if a.x() < pt.x() < b.x() and a.y() < pt.y() < b.y() else None)

    def _drag_crop(self, pt: QPointF):
        kind, (sx, sy), (x0, y0, x1, y1) = self._drag
        h, w, _ = self.linear.shape
        px, py = self._to_photo(pt)
        px, py = min(max(px, 0), w), min(max(py, 0), h)
        if kind == "move":
            dx = min(max(px - sx, -x0), w - x1)
            dy = min(max(py - sy, -y0), h - y1)
            self.set_crop((x0 + dx, y0 + dy, x1 + dx, y1 + dy))
            return
        if kind == "new":
            ax, ay, kind = sx, sy, ("s" if py >= sy else "n") + ("e" if px >= sx else "w")
        else:
            # anchor = the opposite side / corner
            ax = x1 if "w" in kind else x0
            ay = y1 if "n" in kind else y0
        a = self.crop_aspect
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        # size the frame wants, and the room it has from its anchor
        if kind in ("e", "w"):  # one edge: width follows the mouse, height the aspect
            cw, ch = abs(px - ax), (ch0 := y1 - y0)
            room_w, room_h = (ax if kind == "w" else w - ax), 2 * min(cy, h - cy)
            ch = cw / a if a else ch0
        elif kind in ("n", "s"):
            cw, ch = (cw0 := x1 - x0), abs(py - ay)
            room_w, room_h = 2 * min(cx, w - cx), (ay if kind == "n" else h - ay)
            cw = ch * a if a else cw0
        else:  # a corner, or drawing a new frame
            cw, ch = abs(px - ax), abs(py - ay)
            if a:
                cw, ch = (ch * a, ch) if cw < ch * a else (cw, cw / a)
            room_w, room_h = (ax if "w" in kind else w - ax), (ay if "n" in kind else h - ay)
        k = min(1.0, room_w / max(cw, 1e-9), room_h / max(ch, 1e-9))
        cw, ch = cw * k, ch * k
        bx = (ax - cw if "w" in kind else ax) if kind not in ("n", "s") else cx - cw / 2
        by = (ay - ch if "n" in kind else ay) if kind not in ("e", "w") else cy - ch / 2
        box = [bx, by, bx + cw, by + ch]
        if box[2] - box[0] >= self.MIN_CROP and box[3] - box[1] >= self.MIN_CROP:
            self.set_crop(box)

    def set_dev_params(self, params: dict):
        self.dev_params = dict(params)
        self._lut = develop.build_lut(params)
        self._render_view()

    def set_layers(self, dn: np.ndarray | None, sharp: np.ndarray | None):
        if dn is self.dn and sharp is self.sharp:
            return
        self.dn, self.sharp = dn, sharp
        self.ai_updated()

    def set_enh_params(self, params: dict):
        self.enh_params = dict(params)
        if self.mode == "enhance":
            self._render_view()

    def set_mode(self, mode: str):
        self.mode = mode
        self._rebuild_small()

    def ai_updated(self):
        """The AI layers changed (or the worker filled in more tiles)."""
        self._small.pop("dn", None)
        self._small.pop("sharp", None)
        if self.mode == "enhance":
            self._rebuild_small()

    def _target_rect(self) -> QRectF:
        x0, y0, x1, y1 = self._region()
        w, h = x1 - x0, y1 - y0
        s = min(self.width() / w, self.height() / h)
        dw, dh = w * s, h * s
        return QRectF((self.width() - dw) / 2, (self.height() - dh) / 2, dw, dh)

    def _rebuild_small(self):
        if self.linear is None:
            return
        self._rect = self._target_rect()
        dpr = self.devicePixelRatioF()
        size = (max(1, round(self._rect.width() * dpr)), max(1, round(self._rect.height() * dpr)))
        region = self._region()
        if self._small.get("size") != size or self._small.get("region") != region:
            self._small = {"size": size, "region": region}
        x0, y0, x1, y1 = region

        def small(name: str, arr: np.ndarray | None):
            if arr is not None and self._small.get(name) is None:
                self._small[name] = cv2.resize(arr[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)

        small("linear", self.linear)
        if self.mode == "enhance":
            small("dn", self.dn)
            small("sharp", self.sharp)
        self._render_view()

    def _base(self, out_shape, region) -> np.ndarray | None:
        """HDR base layer for a region of the full photo (None = all of it)."""
        if not self.dev_params.get("hdr"):
            return None
        if self._hdr_map is None:
            self._hdr_map = develop.hdr_map(self.linear)
        return develop.hdr_base(self._hdr_map, self.linear.shape, out_shape, region)

    def _render(self, lin, dn, sharp, original: bool, region: tuple | None = None) -> np.ndarray:
        """Render one region (proxy, or loupe crop at `region` = y0, x0, y1, x1) in the current mode."""
        if self.mode == "develop" and original:
            return develop.develop(lin, develop.DEFAULTS, self.baseline_ev, self._default_lut)
        base = self._base(lin.shape, region)
        if self.mode == "enhance" and not original:
            return pipeline.compose(
                lin, dn, sharp, self.g, self.dev_params, self.baseline_ev, self.enh_params, self._lut, base
            )
        return develop.develop(lin, self.dev_params, self.baseline_ev, self._lut, base)

    def _render_view(self):
        """Ask for the view to be re-rendered (in the background, newest state wins)."""
        if "linear" not in self._small:
            return
        self._dirty = True
        if not self._rendering:
            self._start_render()

    def _start_render(self):
        self._dirty = False
        self._rendering = True
        sm = dict(self._small)
        lin, dn, sharp = sm["linear"], sm.get("dn"), sm.get("sharp")
        x0, y0, x1, y1 = sm["region"]
        bw = (lin, dict(self.dev_params), self.baseline_ev) if self.mode == "develop" else None
        self._request.emit((lambda: self._render(lin, dn, sharp, False, (y0, x0, y1, x1)),
                            self.devicePixelRatioF(), bw))

    def _render_done(self, img: QImage, out: np.ndarray, hist, beyond_white: float):
        self._rendering = False
        self._view_img = img
        self._view_orig = None  # made on demand
        self.rendered.emit(out, hist, beyond_white)
        self.update()
        if self._dirty:
            self._start_render()

    def _original_image(self) -> QImage:
        if self._view_orig is None:
            x0, y0, x1, y1 = self._small["region"]
            before = self._render(self._small["linear"], None, None, True, (y0, x0, y1, x1))
            self._view_orig = to_qimage(before, self.devicePixelRatioF())
        return self._view_orig

    def resizeEvent(self, e):
        self._place_loading()
        self._rebuild_small()

    # --- painting -------------------------------------------------------
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 32))
        if self._view_img is None:
            if self.is_loading():
                return
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a photo (Ctrl+O) or drop one here")
            return
        p.drawImage(self._rect.topLeft(), self._original_image() if self.show_orig_all else self._view_img)
        if self.crop_mode:
            self._paint_crop(p)
        elif self.mouse is not None and self._rect.contains(self.mouse):
            self._paint_loupe(p)

    def _paint_crop(self, p: QPainter):
        x0, y0, x1, y1 = self._crop_box()
        box = QRectF(self._to_widget(x0, y0), self._to_widget(x1, y1))
        outside = QPainterPath()
        outside.addRect(self._rect)
        inner = QPainterPath()
        inner.addRect(box)
        p.fillPath(outside.subtracted(inner), QColor(0, 0, 0, 150))
        p.setPen(QPen(QColor(255, 255, 255, 90), 1))
        for i in (1, 2):  # rule of thirds
            x = box.left() + box.width() * i / 3
            y = box.top() + box.height() * i / 3
            p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        p.setPen(QPen(QColor(255, 255, 255), 1.5))
        p.drawRect(box)
        p.setBrush(QColor(255, 255, 255))
        p.setPen(Qt.PenStyle.NoPen)
        for hx in (box.left(), box.center().x(), box.right()):
            for hy in (box.top(), box.center().y(), box.bottom()):
                if (hx, hy) != (box.center().x(), box.center().y()):
                    p.drawRect(QRectF(hx - 4, hy - 4, 8, 8))

    def _paint_loupe(self, p: QPainter):
        dpr = self.devicePixelRatioF()
        h, w, _ = self.linear.shape
        rx0, ry0, rx1, ry1 = self._region()  # the loupe stays inside the crop
        # Image pixel under the cursor
        cx, cy = self._to_photo(self.mouse)
        half = int(self.radius * dpr)
        x0, y0 = int(cx) - half, int(cy) - half
        x1, y1 = x0 + 2 * half, y0 + 2 * half
        # Small margin so the sharpen blur has context at the crop edge
        m = 4
        ex0, ey0, ex1, ey1 = max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, w), min(y1 + m, h)
        ix0, iy0, ix1, iy1 = max(x0, rx0), max(y0, ry0), min(x1, rx1), min(y1, ry1)
        original = self.show_orig_loupe or self.show_orig_all
        region = np.s_[ey0:ey1, ex0:ex1]
        crop = self._render(
            self.linear[region],
            None if self.dn is None else self.dn[region],
            None if self.sharp is None else self.sharp[region],
            original,
            (ey0, ex0, ey1, ex1),
        )
        crop = crop[iy0 - ey0 : iy1 - ey0, ix0 - ex0 : ix1 - ex0]
        if crop.size == 0:
            return
        img = to_qimage(crop, dpr)
        # 1 image pixel = 1 device pixel: the crop's offset from the cursor in logical px
        top_left = QPointF(self.mouse.x() + (ix0 - cx) / dpr, self.mouse.y() + (iy0 - cy) / dpr)

        circle = QPainterPath()
        circle.addEllipse(self.mouse, self.radius, self.radius)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setClipPath(circle)
        p.fillPath(circle, QColor(20, 20, 20))
        p.drawImage(top_left, img)
        p.restore()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255, 200), 2))
        p.drawPath(circle)
        if original:
            p.setPen(QColor(255, 255, 255))
            p.drawText(
                QRectF(self.mouse.x() - 60, self.mouse.y() + self.radius - 26, 120, 20),
                Qt.AlignmentFlag.AlignCenter,
                "original",
            )

    # --- input ----------------------------------------------------------
    CURSORS = {"n": Qt.CursorShape.SizeVerCursor, "s": Qt.CursorShape.SizeVerCursor,
               "e": Qt.CursorShape.SizeHorCursor, "w": Qt.CursorShape.SizeHorCursor,
               "nw": Qt.CursorShape.SizeFDiagCursor, "se": Qt.CursorShape.SizeFDiagCursor,
               "ne": Qt.CursorShape.SizeBDiagCursor, "sw": Qt.CursorShape.SizeBDiagCursor,
               "move": Qt.CursorShape.SizeAllCursor}

    def mouseMoveEvent(self, e):
        self.mouse = e.position()
        if self.crop_mode and self.linear is not None:
            if self._drag:
                self._drag_crop(self.mouse)
            else:
                kind = self._hit(self.mouse)
                self.setCursor(QCursor(self.CURSORS.get(kind, Qt.CursorShape.CrossCursor)))
        self.update()

    def leaveEvent(self, e):
        self.mouse = None
        self.update()

    def mousePressEvent(self, e):
        if self.crop_mode and self.linear is not None:
            if e.button() == Qt.MouseButton.LeftButton:
                kind = self._hit(e.position()) or "new"
                self._drag = (kind, self._to_photo(e.position()), self._crop_box())
            return
        self.show_orig_loupe = True
        self.update()

    def mouseReleaseEvent(self, e):
        if self.crop_mode:
            self._drag = None
            return
        self.show_orig_loupe = False
        self.update()

    def wheelEvent(self, e):
        step = 20 if e.angleDelta().y() > 0 else -20
        self.radius = int(np.clip(self.radius + step, 60, 500))
        self.update()

    def keyPressEvent(self, e):
        if self.crop_mode and e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            self.crop_finished.emit(e.key() != Qt.Key.Key_Escape)
        elif e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self.show_orig_all = True
            self.update()
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self.show_orig_all = False
            self.update()
        else:
            super().keyReleaseEvent(e)


class Histogram(QWidget):
    """Three-channel range view: R, G, B histograms overlaid (additive), with
    markers when a channel clips to black or white."""

    COLORS = [QColor(235, 70, 70), QColor(70, 210, 90), QColor(80, 130, 255)]
    CLIP_WARN = 0.002  # 0.2% of pixels

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.hist: np.ndarray | None = None
        self.clip_lo = self.clip_hi = np.zeros(3)
        self.note = ""

    def set_image(self, rgb: np.ndarray, note: str = ""):
        self.set_histogram(develop.histogram(rgb[::2, ::2]), note)

    def set_histogram(self, hist: tuple, note: str = ""):
        """hist = develop.histogram() result (computed off the GUI thread)."""
        self.hist, self.clip_lo, self.clip_hi = hist
        self.note = note
        self.update()

    def clear(self):
        self.hist = None
        self.note = ""
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        text_h = 34
        plot = QRectF(r.x(), r.y(), r.width(), r.height() - text_h)
        p.fillRect(plot, QColor(18, 18, 20))
        # quarter grid lines: shadows | darks | lights | highlights
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        for i in (1, 2, 3):
            x = plot.x() + plot.width() * i / 4
            p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        if self.hist is None:
            return

        # sqrt scale so small but important tails stay visible; ignore the
        # end bins for scaling (clipped spikes would flatten everything else)
        h = np.sqrt(self.hist.astype(np.float64))
        top = max(h[:, 1:-1].max(), 1e-9)
        bins = h.shape[1]
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        for c in range(3):
            pts = [QPointF(plot.left(), plot.bottom())]
            for i in range(bins):
                x = plot.left() + plot.width() * i / (bins - 1)
                y = plot.bottom() - min(h[c, i] / top, 1.0) * (plot.height() - 4)
                pts.append(QPointF(x, y))
            pts.append(QPointF(plot.right(), plot.bottom()))
            fill = QColor(self.COLORS[c])
            fill.setAlpha(150)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(fill)
            p.drawPolygon(QPolygonF(pts))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # clip markers: a triangle per clipping channel at that end
        for c in range(3):
            for end, frac in (("lo", self.clip_lo[c]), ("hi", self.clip_hi[c])):
                if frac < self.CLIP_WARN:
                    continue
                y = plot.top() + 6 + c * 12
                if end == "lo":
                    tri = [QPointF(plot.left() + 2, y), QPointF(plot.left() + 12, y - 5), QPointF(plot.left() + 12, y + 5)]
                else:
                    tri = [QPointF(plot.right() - 2, y), QPointF(plot.right() - 12, y - 5), QPointF(plot.right() - 12, y + 5)]
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(self.COLORS[c])
                p.drawPolygon(QPolygonF(tri))

        # text: clipping percentages + note
        def pct(v):
            return " ".join(f"{n} {x * 100:.1f}%" for n, x in zip("RGB", v) if x >= self.CLIP_WARN) or "none"

        p.setPen(QColor(160, 160, 160))
        f = p.font()
        f.setPointSizeF(8.5)
        p.setFont(f)
        line1 = f"Clipped black: {pct(self.clip_lo)}   white: {pct(self.clip_hi)}"
        p.drawText(QRectF(r.x(), plot.bottom() + 2, r.width(), 16), Qt.AlignmentFlag.AlignLeft, line1)
        if self.note:
            p.setPen(QColor(230, 190, 90))
            p.drawText(QRectF(r.x(), plot.bottom() + 17, r.width(), 16), Qt.AlignmentFlag.AlignLeft, self.note)
