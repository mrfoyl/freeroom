"""Red eye removal: find the red pupil near a click and neutralise it.

find() builds a soft mask once, from the original photo. apply() runs on any
linear layer (original, denoised, sharpened), at any scale, just before
development, so fixes don't touch the AI results and can be undone.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_RATIO = 1.6  # red must beat green and blue by this much (skin stays around 1.3)
MIN_REDNESS = 0.12
MIN_AREA = 6  # photo pixels
MAX_ASPECT = 2.2  # a pupil is roughly round, also when partly behind an eyelid
MIN_FILL = 0.4  # share of its bounding box (a full circle is 0.79)
MIN_SOLIDITY = 0.75  # share of its outline: pupils are solid, gums and clothes folds aren't
DARKEN = 0.4  # how much the reddest pixels are darkened, so washed-out pupils don't stay grey
GATE_FROM, GATE_TO = 1.05, 1.3  # red/other ratio over which correction fades in: a blue iris stays blue


@dataclass(frozen=True, eq=False)
class Spot:
    x0: int  # top left of the mask, in photo pixels
    y0: int
    mask: np.ndarray  # how much to correct, 0..1
    dark: np.ndarray  # brightness factor of the corrected colour
    cx: float  # pupil centre and radius, for markers and removing
    cy: float
    radius: float


def _perceptual(linear: np.ndarray, baseline_ev: float) -> np.ndarray:
    return np.sqrt(np.clip(linear * np.float32(2.0**baseline_ev), 0.0, 1.0))


def _redness(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gb = np.maximum(e[..., 1], e[..., 2])
    return e[..., 0] - (e[..., 1] + e[..., 2]) / 2, e[..., 0] / np.maximum(gb, 1e-4)


def _search(linear: np.ndarray, x: float, y: float, r: float, reach: float, baseline_ev: float):
    """Best pupil whose centre is within r of (x, y), looking `reach` px around
    it: (labels, stats row, x0, y0), or None."""
    h, w, _ = linear.shape
    x0, y0, x1, y1 = max(int(x - reach), 0), max(int(y - reach), 0), min(int(x + reach) + 1, w), min(int(y + reach) + 1, h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    # a little blur, so sensor noise doesn't break the pupil into pieces
    e = cv2.GaussianBlur(_perceptual(linear[y0:y1, x0:x1], baseline_ev), (0, 0), max(0.7, r / 80))
    redness, ratio = _redness(e)
    red = ((ratio > MIN_RATIO) & (redness > MIN_REDNESS)).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(red, connectivity=8)

    best, best_score = None, 0.0
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        # touching the edge: bigger than this window (skin, lips, clothes…, or a pupil for a wider look)
        touches = (bx == 0 and x0 > 0) or (by == 0 and y0 > 0) or (bx + bw == x1 - x0 and x1 < w) or (by + bh == y1 - y0 and y1 < h)
        if touches or area < MIN_AREA or max(bw, bh) > MAX_ASPECT * min(bw, bh) or area < MIN_FILL * bw * bh:
            continue
        contours, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if area < MIN_SOLIDITY * cv2.contourArea(cv2.convexHull(np.vstack(contours))):
            continue
        dist = np.hypot(centroids[i][0] + x0 - x, centroids[i][1] + y0 - y)
        if dist > r:
            continue
        score = float(redness[labels == i].mean()) * area * (area / (bw * bh)) / (1 + 2 * dist / r)
        if score > best_score:
            best, best_score = i, score
    return None if best is None else (labels == best, stats[best], x0, y0)


def find(linear: np.ndarray, x: float, y: float, r: float, baseline_ev: float = 0.0) -> Spot | None:
    """The red pupil whose centre is within r photo pixels of (x, y), or None."""
    h, w, _ = linear.shape
    # Look wider if needed: a pupil may reach past the circle when the click is off-centre.
    for reach in (r * 1.25, r * 2, r * 3.5):
        found = _search(linear, x, y, r, reach, baseline_ev)
        if found is not None:
            break
    else:
        return None
    blob, stats, x0, y0 = found

    # Filled outline: the catchlight inside the pupil is part of the shape, and
    # stays white by itself because correcting a neutral colour changes nothing.
    contours, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(contours))
    (cx, cy), radius = cv2.minEnclosingCircle(hull)
    # Reach past the detected edge: its transition pixels are still reddish. The
    # colour gate below keeps this from touching the iris around the pupil.
    grow = int(np.ceil(max(2.0, radius * 0.2)))
    sigma = max(0.8, radius * 0.06)  # soft edge
    pad = grow + int(np.ceil(3 * sigma)) + 2
    bx, by, bw, bh, _ = stats
    mx0, my0 = max(x0 + bx - pad, 0), max(y0 + by - pad, 0)
    mx1, my1 = min(x0 + bx + bw + pad, w), min(y0 + by + bh + pad, h)
    mask = np.zeros((my1 - my0, mx1 - mx0), np.uint8)
    cv2.fillConvexPoly(mask, hull + (x0 - mx0, y0 - my0), 1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)

    # lightly smoothed like the detection, so noise doesn't speckle the gate
    area_e = cv2.GaussianBlur(_perceptual(linear[my0:my1, mx0:mx1], baseline_ev), (0, 0), 0.7)
    area_redness, area_ratio = _redness(area_e)
    mask = mask * np.clip((area_ratio - GATE_FROM) / (GATE_TO - GATE_FROM), 0.0, 1.0)
    dark = 1.0 - DARKEN * np.clip(area_redness / 0.4, 0.0, 1.0)
    return Spot(mx0, my0, mask, dark.astype(np.float32), float(cx + x0), float(cy + y0), float(radius))


def apply(x: np.ndarray, spots, region: tuple | None = None) -> np.ndarray:
    """Copy of linear layer x with the spots corrected. x shows `region`
    (y0, x0, y1, x1 in photo pixels), at any scale; None = x is the whole photo at full size."""
    if not spots:
        return x
    h, w, _ = x.shape
    ry0, rx0, ry1, rx1 = region or (0, 0, h, w)
    sx, sy = w / (rx1 - rx0), h / (ry1 - ry0)
    out = None
    for s in spots:
        sh, sw = s.mask.shape
        ax0, ay0 = round((s.x0 - rx0) * sx), round((s.y0 - ry0) * sy)
        ax1, ay1 = round((s.x0 + sw - rx0) * sx), round((s.y0 + sh - ry0) * sy)
        ax1, ay1 = max(ax1, ax0 + 1), max(ay1, ay0 + 1)
        if ax1 <= 0 or ay1 <= 0 or ax0 >= w or ay0 >= h:
            continue
        mask, dark = s.mask, s.dark
        if (ax1 - ax0, ay1 - ay0) != (sw, sh):
            mask = cv2.resize(mask, (ax1 - ax0, ay1 - ay0), interpolation=cv2.INTER_AREA)
            dark = cv2.resize(dark, (ax1 - ax0, ay1 - ay0), interpolation=cv2.INTER_AREA)
        cx0, cy0, cx1, cy1 = max(ax0, 0), max(ay0, 0), min(ax1, w), min(ay1, h)
        cut = np.s_[cy0 - ay0 : cy1 - ay0, cx0 - ax0 : cx1 - ax0]
        if out is None:
            out = x.copy()
        area = out[cy0:cy1, cx0:cx1]
        target = (area[..., 1] + area[..., 2]) / 2 * dark[cut]
        area += mask[cut][..., None] * (target[..., None] - area)
    return x if out is None else out
