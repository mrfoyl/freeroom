"""Layers: a selection (mask) plus a full set of settings that applies inside it.

The Base layer is the normal photo; each layer above it is developed with its
own settings (Raw, B&W, Enhancement) and blended in where its mask is. Masks
live at the work resolution (segment.WORK) and are scaled to whatever is
being rendered: the screen-size view, the loupe or the saved file.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

_ids = itertools.count(1)
MAX_UNDO = 20


@dataclass
class Layer:
    name: str
    settings: dict  # same form as MainWindow.settings()
    work: tuple[int, int]  # mask size (h, w)
    full: tuple[int, int]  # photo size (h, w)
    uid: int = field(default_factory=lambda: next(_ids))
    visible: bool = True
    invert: bool = False
    feather: float = 0.0  # blur of the mask edge, photo px
    opacity: float = 1.0
    points: list = field(default_factory=list)  # AI prompts: (x, y, label) in photo px
    ai: np.ndarray | None = None  # AI selection (float32, work size)
    paint: np.ndarray | None = None  # brush strokes: +1 keep, -1 remove (int8, work size)
    mask: np.ndarray | None = None  # final mask (float32 0..1, work size)
    undo: list = field(default_factory=list)

    def __post_init__(self):
        if self.paint is None:
            self.paint = np.zeros(self.work, np.int8)
        self.rebuild()

    @property
    def scale(self) -> float:
        return self.work[1] / self.full[1]

    def snapshot(self):
        """Before a selection change, for Undo."""
        self.undo.append((list(self.points), None if self.ai is None else self.ai.copy(), self.paint.copy()))
        del self.undo[:-MAX_UNDO]

    def undo_last(self) -> bool:
        if not self.undo:
            return False
        self.points, self.ai, self.paint = self.undo.pop()
        self.rebuild()
        return True

    def clear(self):
        self.snapshot()
        self.points, self.ai = [], None
        self.paint[:] = 0
        self.rebuild()

    def stroke(self, pts: list[tuple[float, float]], radius: float, keep: bool):
        """Paint a brush stroke (photo px) into the mask: keep or remove."""
        k = self.scale
        r = max(1, round(radius * k))
        xy = [(round(x * k), round(y * k)) for x, y in pts]
        val = 1 if keep else -1
        layer = np.zeros(self.work, np.uint8)
        for a, b in zip(xy, xy[1:] or xy):
            cv2.line(layer, a, b, 1, thickness=2 * r, lineType=cv2.LINE_AA)
        self.paint[layer > 0] = val

    def rebuild(self):
        m = np.zeros(self.work, np.float32) if self.ai is None else self.ai.copy()
        m[self.paint > 0] = 1.0
        m[self.paint < 0] = 0.0
        if self.feather > 0:
            m = cv2.GaussianBlur(m, (0, 0), max(0.3, self.feather * self.scale))
        if self.invert:
            m = 1.0 - m
        self.mask = (m * self.opacity).astype(np.float32)

    def empty(self) -> bool:
        return self.mask is None or float(self.mask.max()) <= 1e-3

    def copy_settings(self) -> dict:
        return copy.deepcopy(self.settings)


def region_mask(mask: np.ndarray, full: tuple[int, int], region: tuple[int, int, int, int],
                out_hw: tuple[int, int]) -> np.ndarray:
    """The mask for photo region (y0, x0, y1, x1), resampled to out_hw (h, w)."""
    y0, x0, y1, x1 = region
    oh, ow = out_hw
    sy, sx = mask.shape[0] / full[0], mask.shape[1] / full[1]
    # output pixel (u, v) centre -> photo -> mask pixel
    ax, ay = (x1 - x0) / ow * sx, (y1 - y0) / oh * sy
    bx, by = x0 * sx + 0.5 * ax - 0.5, y0 * sy + 0.5 * ay - 0.5
    m = np.float32([[ax, 0, bx], [0, ay, by]])
    src = mask
    if ax > 2 or ay > 2:  # shrinking a lot: pre-blur so it doesn't alias
        src = cv2.GaussianBlur(mask, (0, 0), max(ax, ay) / 2.5)
    return cv2.warpAffine(src, m, (ow, oh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def blend(below: np.ndarray, above: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return below + (above - below) * mask[..., None]
