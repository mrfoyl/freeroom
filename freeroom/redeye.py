"""Red eye removal: find red pupils and neutralise them.

detect() looks through the whole photo: red round blobs are paired by the rule
of a pair of eyes (same size, and as far apart as a face has them, measured in
pupil diameters, to either side), then a face detector (YuNet, which comes with
OpenCV) has to confirm that a face's eyes are on them. So two tail lights or a
red pattern aren't "eyes". find() does the same for one click, partner() finds
the other eye from the one that was found.

find() builds a soft mask once, from the original photo. apply() runs on any
linear layer (original, denoised, sharpened), at any scale, just before
development, so fixes don't touch the AI results and can be undone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

# A pair of eyes. Faces are about 63 mm between the pupils and pupils 2.5-8 mm
# wide, though the red part can be smaller or, with a bright rim, a little bigger.
# Dark maroon pupils often have only a bright red fragment. From there the mask
# grows over what's still blood-coloured: green below blue keeps it out of brown irises and skin.
GROW_RATIO, GROW_REDNESS, GROW_GREEN_BLUE = 1.25, 0.03, 1.1
GROW_REACH = 3.5  # how far to look, in sizes of the fragment
GROW_ROUND = 0.6  # share of its enclosing circle: anything less has run out over an eyelid
GROW_DARK = 1.3, 0.02  # green and blue may be this much (times, plus) above the fragment's: a pupil is dark in them, skin isn't

MIN_SPACING, MAX_SPACING = 5.0, 32.0  # distance between the eyes, in pupil diameters
MIN_SIZE_MATCH = 0.55  # smaller pupil / bigger pupil
MAX_TILT = 40.0  # degrees from level for a partner found from a click (head tilted)
MAX_PAIR_TILT = 90.0  # any angle for pairs a face detector still has to confirm (sideways photo)

FACE_MODEL = Path(__file__).parent / "data" / "face_detection_yunet_2026may.onnx"
FACE_SIZE = 320  # px of the crop shown to the detector
FACE_EYES = 72.0  # px between the eyes in that crop
FACE_SCORE = 0.45
SURE_RATIO = 1.9  # red pupils measured 2.0-2.9 in real photos, skin and brown irises up to 1.76 ...
SURE_GREEN_BLUE = 1.1  # ... and blood has green below blue (0.7-0.9), skin above it (1.0-2.0)
EYE_TOLERANCE = 0.15  # how far a detected eye may be from a blob, in eye distances
SINGLE_GUESSES = (9.0, 14.0, 22.0)  # eye distances to try for a lone blob, in pupil diameters
MAX_BLOBS = 60  # the most promising ones; a red pattern can have thousands
MAX_CHECKS = 80  # face detector runs per photo


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


def _grow(linear: np.ndarray, blob: np.ndarray, stats, x0: int, y0: int, baseline_ev: float):
    """The whole pupil around a fragment of it, in the same form as _search
    gives, or None if what's around it doesn't have the shape of a pupil."""
    h, w, _ = linear.shape
    bx, by, bw, bh, _ = stats
    reach = int(np.ceil(max(bw, bh) * GROW_REACH))
    wx0, wy0 = max(x0 + bx - reach, 0), max(y0 + by - reach, 0)
    wx1, wy1 = min(x0 + bx + bw + reach, w), min(y0 + by + bh + reach, h)
    e = cv2.GaussianBlur(_perceptual(linear[wy0:wy1, wx0:wx1], baseline_ev), (0, 0), 0.8)
    redness, ratio = _redness(e)
    gb = np.maximum(e[..., 1], e[..., 2])
    ys, xs = np.nonzero(blob)
    ys, xs = ys + y0 - wy0, xs + x0 - wx0
    dark = gb < np.median(gb[ys, xs]) * GROW_DARK[0] + GROW_DARK[1]
    low = (ratio > GROW_RATIO) & (redness > GROW_REDNESS) & dark & (e[..., 1] / np.maximum(e[..., 2], 1e-4) < GROW_GREEN_BLUE)
    low[ys, xs] = True
    py, px = ys[0], xs[0]
    _, labels, st, _ = cv2.connectedComponentsWithStats(low.astype(np.uint8), connectivity=8)
    i = labels[py, px]
    lx, ly, lw, lh, area = st[i]
    touches = (lx == 0 and wx0 > 0) or (ly == 0 and wy0 > 0) or (lx + lw == wx1 - wx0 and wx1 < w) or (ly + lh == wy1 - wy0 and wy1 < h)
    if touches or max(lw, lh) > MAX_ASPECT * min(lw, lh) or area < MIN_FILL * lw * lh:
        return None
    contours, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(contours))
    _, radius = cv2.minEnclosingCircle(hull)
    if area < MIN_SOLIDITY * cv2.contourArea(hull) or area < GROW_ROUND * np.pi * radius**2:
        return None
    return labels == i, st[i], wx0, wy0


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
    blob, stats, x0, y0 = _grow(linear, *found, baseline_ev) or found

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


@dataclass(frozen=True)
class _Blob:
    x: float
    y: float
    radius: float
    weight: float  # how much red there is, for picking the most promising


def _blobs(linear: np.ndarray, baseline_ev: float, ox: int = 0, oy: int = 0) -> list[_Blob]:
    """Every clearly red, round, solid blob in this part of the photo (at offset
    ox, oy). Stricter than a click: nobody pointed at these, so reddish skin,
    brows and brown irises must stay out."""
    e = cv2.GaussianBlur(_perceptual(linear, baseline_ev), (0, 0), 0.8)
    redness, ratio = _redness(e)
    red = ((ratio > MIN_RATIO) & (redness > MIN_REDNESS)).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(red, connectivity=8)
    found = []
    longest = max(linear.shape[:2]) / 8  # no pupil is this big
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < MIN_AREA or max(bw, bh) > longest or max(bw, bh) > MAX_ASPECT * min(bw, bh) or area < MIN_FILL * bw * bh:
            continue
        inside = labels[by : by + bh, bx : bx + bw] == i
        box = e[by : by + bh, bx : bx + bw][inside]
        if np.median(ratio[by : by + bh, bx : bx + bw][inside]) < SURE_RATIO:
            continue
        if np.median(box[:, 1] / np.maximum(box[:, 2], 1e-4)) > SURE_GREEN_BLUE:
            continue
        contours, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack(contours))
        if area < MIN_SOLIDITY * cv2.contourArea(hull):
            continue
        _, radius = cv2.minEnclosingCircle(hull)
        found.append(_Blob(float(centroids[i][0] + ox), float(centroids[i][1] + oy), max(float(radius), 1.0), float(area * redness[by : by + bh, bx : bx + bw][inside].mean())))
    return found


def _pair_fits(a: _Blob, b: _Blob, max_tilt: float) -> bool:
    """Same size, and as far apart as eyes are for pupils of that size."""
    if min(a.radius, b.radius) < MIN_SIZE_MATCH * max(a.radius, b.radius):
        return False
    dx, dy = abs(a.x - b.x), abs(a.y - b.y)
    spacing = np.hypot(dx, dy) / (a.radius + b.radius)
    return MIN_SPACING <= spacing <= MAX_SPACING and np.degrees(np.arctan2(dy, max(dx, 1e-6))) <= max_tilt


def partner(linear: np.ndarray, spot: Spot, baseline_ev: float = 0.0) -> Spot | None:
    """The other eye of the pupil in `spot`: looks to both sides of it for the
    same size of red pupil, at the distance eyes have for a pupil that size."""
    h, w, _ = linear.shape
    me = _Blob(spot.cx, spot.cy, spot.radius, 0.0)
    reach_x = 2 * spot.radius * MAX_SPACING / MIN_SIZE_MATCH
    reach_y = reach_x * np.sin(np.radians(MAX_TILT))
    x0, y0 = max(int(spot.cx - reach_x), 0), max(int(spot.cy - reach_y), 0)
    x1, y1 = min(int(spot.cx + reach_x) + 1, w), min(int(spot.cy + reach_y) + 1, h)
    fits = [b for b in _blobs(linear[y0:y1, x0:x1], baseline_ev, x0, y0)
            if np.hypot(b.x - me.x, b.y - me.y) > me.radius + b.radius and _pair_fits(me, b, MAX_TILT)]

    def misfit(b: _Blob) -> float:  # most alike and most level first, then the nearest
        tilt = np.degrees(np.arctan2(abs(b.y - me.y), abs(b.x - me.x) + 1e-6))
        return abs(np.log(b.radius / me.radius)) * 3 + tilt / 30 + np.hypot(b.x - me.x, b.y - me.y) / reach_x
    for b in sorted(fits, key=misfit):
        found = find(linear, b.x, b.y, max(b.radius * 1.5, 6.0), baseline_ev)
        if found is not None:
            return found
    return None


class _Faces:
    """YuNet on a crop around a point, with the results back in photo pixels."""

    def __init__(self, linear: np.ndarray, baseline_ev: float):
        self.linear, self.ev = linear, baseline_ev
        self.detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (FACE_SIZE, FACE_SIZE), FACE_SCORE, 0.3, 20)
        self.checks = 0

    def eyes(self, x: float, y: float, eye_distance: float, angle: float = 0.0) -> list[np.ndarray]:
        """Eye pairs (2x2: x, y of each, photo pixels) of the faces in a crop where
        (x, y) is between the eyes, they're `eye_distance` apart and `angle` degrees off level."""
        self.checks += 1
        h, w, _ = self.linear.shape
        k = FACE_EYES / eye_distance
        # shrink nicely first: warpAffine alone would alias at a small k
        half = int(np.ceil(FACE_SIZE / k))
        x0, y0, x1, y1 = max(int(x) - half, 0), max(int(y) - half, 0), min(int(x) + half, w), min(int(y) + half, h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return []
        area = self.linear[y0:y1, x0:x1]
        shrink = min(1.0, k * 1.5)
        if shrink < 1.0:
            area = cv2.resize(area, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
        fx, fy = area.shape[1] / (x1 - x0), area.shape[0] / (y1 - y0)
        m = cv2.getRotationMatrix2D(((x - x0) * fx, (y - y0) * fy), angle, k / fx)
        m[0, 2] += FACE_SIZE / 2 - (x - x0) * fx
        m[1, 2] += FACE_SIZE * 0.4 - (y - y0) * fy
        crop = cv2.warpAffine(area, m, (FACE_SIZE, FACE_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        image = (_perceptual(crop, self.ev)[..., ::-1] * 255).astype(np.uint8)
        _, faces = self.detector.detect(image)
        if faces is None:
            return []
        back = np.linalg.inv(np.vstack([m, (0, 0, 1)]))[:2]
        pairs = []
        for f in faces:
            eyes = np.c_[f[4:8].reshape(2, 2), np.ones(2)] @ back.T  # to the shrunk area
            pairs.append(eyes / (fx, fy) + (x0, y0))
        return pairs

    def confirm_pair(self, a: _Blob, b: _Blob) -> bool:
        d = float(np.hypot(a.x - b.x, a.y - b.y))
        angle = float(np.degrees(np.arctan2(b.y - a.y, b.x - a.x)))
        for turn in (angle, angle + 180.0):  # the top of the head is on an unknown side
            for eyes in self.eyes((a.x + b.x) / 2, (a.y + b.y) / 2, d, turn):
                tol = EYE_TOLERANCE * d
                ends = np.array([[a.x, a.y], [b.x, b.y]])
                if (np.hypot(*(eyes - ends).T).max() < tol) or (np.hypot(*(eyes - ends[::-1]).T).max() < tol):
                    return True
        return False

    def confirm_single(self, a: _Blob) -> bool:
        """A lone red pupil (the other eye turned away or not red): a face's eye has
        to be on it, and that face's eye distance has to suit a pupil of this size."""
        for spacing in SINGLE_GUESSES:
            guess = spacing * 2 * a.radius
            for side in (1, -1):  # it may be either eye
                for eyes in self.eyes(a.x + side * guess / 2, a.y, guess):
                    d = float(np.hypot(*(eyes[0] - eyes[1])))
                    near = np.hypot(*(eyes - (a.x, a.y)).T).min()
                    if near < EYE_TOLERANCE * d and MIN_SPACING <= d / (2 * a.radius) <= MAX_SPACING:
                        return True
        return False


def detect(linear: np.ndarray, baseline_ev: float = 0.0) -> list[Spot]:
    """Red pupils of the whole photo: pairs first, then any lone ones on a face."""
    blobs = sorted(_blobs(linear, baseline_ev), key=lambda b: -b.weight)[:MAX_BLOBS]
    if not blobs:
        return []
    faces = _Faces(linear, baseline_ev)
    pairs = [(a, b) for i, a in enumerate(blobs) for b in blobs[i + 1 :] if _pair_fits(a, b, MAX_PAIR_TILT)]
    pairs.sort(key=lambda p: -(p[0].weight + p[1].weight))
    eyes: list[_Blob] = []
    for a, b in pairs:
        if a in eyes or b in eyes or faces.checks >= MAX_CHECKS:
            continue
        if faces.confirm_pair(a, b):
            eyes += [a, b]
    for a in blobs:
        if a not in eyes and faces.checks < MAX_CHECKS and faces.confirm_single(a):
            eyes.append(a)
    spots = (find(linear, b.x, b.y, max(b.radius * 1.5, 6.0), baseline_ev) for b in eyes)
    return [s for s in spots if s is not None]


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
