"""Photo view with the 100% loupe, and the RGB range view (histogram)."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from . import develop, engine


def to_qimage(rgb: np.ndarray, dpr: float = 1.0) -> QImage:
    arr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    h, w, _ = arr.shape
    img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    img.setDevicePixelRatio(dpr)
    return img


class PhotoView(QWidget):
    """Fit-to-window preview with a circular 100% loupe under the mouse.

    Two modes, one per tab:
      "develop": developed RAW, computed on the fly from linear data
                 (original = default development)
      "enhance": developed photo + AI enhancement (original = developed photo)
    """

    rendered = Signal(object)  # screen-size result, for the histogram

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 300)
        self.mode = "develop"
        # develop mode
        self.linear: np.ndarray | None = None
        self.baseline_ev = 0.0
        self.dev_params = dict(develop.DEFAULTS)
        self._lut = self._default_lut = develop.build_lut(develop.DEFAULTS)
        # enhance mode
        self.orig: np.ndarray | None = None  # developed
        self.ai: np.ndarray | None = None
        self.sharp: np.ndarray | None = None
        self.enh_params: dict = {}
        # view state
        self.radius = 160
        self.mouse: QPointF | None = None
        self.show_orig_loupe = False  # mouse button held
        self.show_orig_all = False  # space held
        self._small: dict[str, np.ndarray | None] = {}
        self._view_img: QImage | None = None
        self._view_orig: QImage | None = None
        self._rect = QRectF()

    # --- data -----------------------------------------------------------
    def set_linear(self, linear: np.ndarray, baseline_ev: float):
        self.linear, self.baseline_ev = linear, baseline_ev
        self.orig = self.ai = self.sharp = None
        self._small = {}
        self._rebuild_small()

    def set_dev_params(self, params: dict):
        self.dev_params = dict(params)
        self._lut = develop.build_lut(params)
        if self.mode == "develop":
            self._render_view()

    def set_enhance(self, orig: np.ndarray, ai: np.ndarray, sharp: np.ndarray | None):
        self.orig, self.ai, self.sharp = orig, ai, sharp
        for k in ("orig", "ai", "sharp"):
            self._small.pop(k, None)
        if self.mode == "enhance":
            self._rebuild_small()

    def set_sharp(self, sharp: np.ndarray | None):
        self.sharp = sharp
        self._small.pop("sharp", None)
        if self.mode == "enhance":
            self._rebuild_small()

    def set_enh_params(self, params: dict):
        self.enh_params = dict(params)
        if self.mode == "enhance":
            self._render_view()

    def set_mode(self, mode: str):
        self.mode = mode
        self._rebuild_small()

    def ai_updated(self):
        """The worker filled more tiles into ai/sharp."""
        self._small.pop("ai", None)
        self._small.pop("sharp", None)
        if self.mode == "enhance":
            self._rebuild_small()

    def proxy_linear(self) -> np.ndarray | None:
        """The screen-size linear photo (cheap stand-in for statistics)."""
        return self._small.get("linear")

    def _enhance_ready(self) -> bool:
        return self.mode == "enhance" and self.orig is not None and self.ai is not None

    def _target_rect(self) -> QRectF:
        h, w, _ = self.linear.shape
        s = min(self.width() / w, self.height() / h)
        dw, dh = w * s, h * s
        return QRectF((self.width() - dw) / 2, (self.height() - dh) / 2, dw, dh)

    def _rebuild_small(self):
        if self.linear is None:
            return
        self._rect = self._target_rect()
        dpr = self.devicePixelRatioF()
        size = (max(1, round(self._rect.width() * dpr)), max(1, round(self._rect.height() * dpr)))
        if self._small.get("size") != size:
            self._small = {"size": size}

        def small(name: str, arr: np.ndarray | None):
            if arr is not None and self._small.get(name) is None:
                self._small[name] = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)

        small("linear", self.linear)
        if self._enhance_ready():
            small("orig", self.orig)
            small("ai", self.ai)
            small("sharp", self.sharp)
        self._render_view()

    def _render_view(self):
        if "linear" not in self._small:
            return
        dpr = self.devicePixelRatioF()
        if self._enhance_ready() and "orig" in self._small:
            out = engine.adjust(
                self._small["orig"], self._small["ai"], self._small.get("sharp"), **self.enh_params
            )
        else:
            out = develop.develop(self._small["linear"], self.dev_params, self.baseline_ev, self._lut)
        self._view_img = to_qimage(out, dpr)
        self._view_orig = None  # made on demand
        self.rendered.emit(out)
        self.update()

    def _original_image(self) -> QImage:
        if self._view_orig is None:
            if self._enhance_ready() and "orig" in self._small:
                before = self._small["orig"]
            else:
                before = develop.develop(self._small["linear"], develop.DEFAULTS, self.baseline_ev, self._default_lut)
            self._view_orig = to_qimage(before, self.devicePixelRatioF())
        return self._view_orig

    def resizeEvent(self, e):
        self._rebuild_small()

    # --- painting -------------------------------------------------------
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 32))
        if self._view_img is None:
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a photo (Ctrl+O) or drop one here")
            return
        p.drawImage(self._rect.topLeft(), self._original_image() if self.show_orig_all else self._view_img)
        if self.mouse is not None and self._rect.contains(self.mouse):
            self._paint_loupe(p)

    def _loupe_crop(self, y0, y1, x0, x1, m, original: bool) -> np.ndarray:
        """Rendered pixels for [y0:y1, x0:x1] (already including margin m)."""
        if self._enhance_ready():
            o = self.orig[y0:y1, x0:x1]
            if original:
                return o
            sh = None if self.sharp is None else self.sharp[y0:y1, x0:x1]
            return engine.adjust(o, self.ai[y0:y1, x0:x1], sh, **self.enh_params)
        lin = self.linear[y0:y1, x0:x1]
        if original:
            return develop.develop(lin, develop.DEFAULTS, self.baseline_ev, self._default_lut)
        return develop.develop(lin, self.dev_params, self.baseline_ev, self._lut)

    def _paint_loupe(self, p: QPainter):
        dpr = self.devicePixelRatioF()
        h, w, _ = self.linear.shape
        # Image pixel under the cursor
        cx = (self.mouse.x() - self._rect.x()) / self._rect.width() * w
        cy = (self.mouse.y() - self._rect.y()) / self._rect.height() * h
        half = int(self.radius * dpr)
        x0, y0 = int(cx) - half, int(cy) - half
        x1, y1 = x0 + 2 * half, y0 + 2 * half
        # Small margin so the sharpen blur has context at the crop edge
        m = 4
        ex0, ey0, ex1, ey1 = max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, w), min(y1 + m, h)
        ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        original = self.show_orig_loupe or self.show_orig_all
        crop = self._loupe_crop(ey0, ey1, ex0, ex1, m, original)
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
    def mouseMoveEvent(self, e):
        self.mouse = e.position()
        self.update()

    def leaveEvent(self, e):
        self.mouse = None
        self.update()

    def mousePressEvent(self, e):
        self.show_orig_loupe = True
        self.update()

    def mouseReleaseEvent(self, e):
        self.show_orig_loupe = False
        self.update()

    def wheelEvent(self, e):
        step = 20 if e.angleDelta().y() > 0 else -20
        self.radius = int(np.clip(self.radius + step, 60, 500))
        self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
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
        self.hist, self.clip_lo, self.clip_hi = develop.histogram(rgb[::2, ::2])
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
