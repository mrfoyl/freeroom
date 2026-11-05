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
SOFTNESS = 0.3  # mask edge (sigma) per pixel an edge around the eye takes
PUPIL_MAX = 0.25  # the brightest a fixed pupil gets, perceptual: blur can leave it pale
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
WHOLE_SIZE = 1280  # px, long side, for the first look at the whole photo
EYE_NEAR = 0.2  # how far a red blob may be from a face's eye, in eye distances
FACE_SIZE = 320  # px of the crop shown to the detector
FACE_EYES = 72.0  # px between the eyes in that crop
FACE_SCORE = 0.45
SURE_RATIO = 1.9  # red pupils measured 2.0-2.9 in real photos, skin and brown irises up to 1.76 ...
SURE_GREEN_BLUE = 1.1  # ... and blood has green below blue (0.7-0.9), skin above it (1.0-2.0)
# Blur mixes a pupil's red with what's around it and smears its shape. Such a blob
# isn't enough by itself, but is as the partner of a sure eye, or as a pair on a face.
BLUR_RATIO = 1.65
BLUR_GREEN_BLUE = 0.92  # measured on real photos: lashes, lips and eye corners have 0.95-1.0 here ...
BLUR_BRIGHT_RED = 0.3  # ... and weak red (peak up to 0.21); a pupil mixed with white can be 0.96 but is bright (0.43)
BLUR_CONTRAST = 1.15  # redder than right around it by this factor: reddish skin isn't
BLUR_ASPECT, BLUR_FILL, BLUR_SOLIDITY = 3.5, 0.35, 0.65
# A last look at each fix, measured on real photos. Red this dim is mostly a black patch
# waiting to happen in a dark maroon pupil, or a spot on a face the detector imagined.
DIM_RED = 0.17
# Noise and eyelids: not that red, green about level with blue, and not darker than around them.
WEAK_RATIO, WEAK_GREEN_BLUE, WEAK_DARKNESS, WEAK_PEAK = 1.8, 0.93, 0.7, 0.35
BLUR_EYE_TOLERANCE = 0.3  # the detector places eyes less exactly in a blurred face
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


def _colour(linear: np.ndarray, baseline_ev: float) -> np.ndarray:
    """For measuring colour: no ceiling, or a RAW's base exposure would flatten bright red into white."""
    return np.sqrt(np.maximum(linear * np.float32(2.0**baseline_ev), 0.0))


def _redness(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gb = np.maximum(e[..., 1], e[..., 2])
    return e[..., 0] - (e[..., 1] + e[..., 2]) / 2, e[..., 0] / np.maximum(gb, 1e-4)


def _search(linear: np.ndarray, x: float, y: float, r: float, reach: float, baseline_ev: float, blurred: bool = False):
    """Best pupil whose centre is within r of (x, y), looking `reach` px around
    it: (labels, stats row, x0, y0), or None. blurred = allow a smeared shape."""
    max_aspect, min_fill, min_solidity = (BLUR_ASPECT, BLUR_FILL, BLUR_SOLIDITY) if blurred else (MAX_ASPECT, MIN_FILL, MIN_SOLIDITY)
    h, w, _ = linear.shape
    x0, y0, x1, y1 = max(int(x - reach), 0), max(int(y - reach), 0), min(int(x + reach) + 1, w), min(int(y + reach) + 1, h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    # a little blur, so sensor noise doesn't break the pupil into pieces
    e = cv2.GaussianBlur(_colour(linear[y0:y1, x0:x1], baseline_ev), (0, 0), max(0.7, r / 80))
    redness, ratio = _redness(e)
    red = ((ratio > MIN_RATIO) & (redness > MIN_REDNESS)).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(red, connectivity=8)

    best, best_score = None, 0.0
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        # touching the edge: bigger than this window (skin, lips, clothes…, or a pupil for a wider look)
        touches = (bx == 0 and x0 > 0) or (by == 0 and y0 > 0) or (bx + bw == x1 - x0 and x1 < w) or (by + bh == y1 - y0 and y1 < h)
        if touches or area < MIN_AREA or max(bw, bh) > max_aspect * min(bw, bh) or area < min_fill * bw * bh:
            continue
        contours, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if area < min_solidity * cv2.contourArea(cv2.convexHull(np.vstack(contours))):
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
    e = cv2.GaussianBlur(_colour(linear[wy0:wy1, wx0:wx1], baseline_ev), (0, 0), 0.8)
    redness, ratio = _redness(e)
    gb = np.maximum(e[..., 1], e[..., 2])
    ys, xs = np.nonzero(blob)
    ys, xs = ys + y0 - wy0, xs + x0 - wx0
    dark = gb < np.median(gb[ys, xs]) * GROW_DARK[0] + GROW_DARK[1]
    low = (ratio > GROW_RATIO) & (redness > GROW_REDNESS) & dark & (e[..., 1] / np.maximum(e[..., 2], 1e-4) < GROW_GREEN_BLUE)
    low[ys, xs] = True
    py, px = ys[0], xs[0]
    # a catchlight smeared into a ring cuts the pupil in pieces: bridge it
    size = 2 * (max(bw, bh) // 3) + 3
    low = cv2.morphologyEx(low.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
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


def _softness(linear: np.ndarray, x: int, y: int, bw: int, bh: int, radius: float, baseline_ev: float) -> float:
    """How soft the mask's edge has to be (sigma, px): as soft as the eye's
    surroundings, or a fix in a blurred photo looks pasted on. A wide look, because
    the catchlight can stay sharp in a blurred eye (a flash freezes it)."""
    pad = int(np.ceil(radius * 3)) + 3
    area = _perceptual(linear[max(y - pad, 0) : y + bh + pad, max(x - pad, 0) : x + bw + pad], baseline_ev)[..., 1]
    area = cv2.GaussianBlur(area, (0, 0), 0.8)
    slope = np.hypot(cv2.Sobel(area, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(area, cv2.CV_32F, 0, 1, ksize=3)) / 8
    ay, ax = np.indices(area.shape)
    around = np.hypot(ax - (x + bw / 2 - max(x - pad, 0)), ay - (y + bh / 2 - max(y - pad, 0))) > radius * 1.4
    low, high = np.percentile(area[around], (5, 95))
    width = float(high - low) / max(float(np.percentile(slope[around], 99)), 1e-4)  # px the steepest edges take
    return float(np.clip(width * SOFTNESS, max(0.8, radius * 0.06), radius * 0.6))


def find(linear: np.ndarray, x: float, y: float, r: float, baseline_ev: float = 0.0, blurred: bool = False) -> Spot | None:
    """The red pupil whose centre is within r photo pixels of (x, y), or None."""
    h, w, _ = linear.shape
    # Look wider if needed: a pupil may reach past the circle when the click is off-centre.
    for reach in (r * 1.25, r * 2, r * 3.5):
        found = _search(linear, x, y, r, reach, baseline_ev, blurred)
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
    sigma = _softness(linear, x0 + stats[0], y0 + stats[1], stats[2], stats[3], radius, baseline_ev)
    pad = grow + int(np.ceil(3 * sigma)) + 2
    bx, by, bw, bh, _ = stats
    mx0, my0 = max(x0 + bx - pad, 0), max(y0 + by - pad, 0)
    mx1, my1 = min(x0 + bx + bw + pad, w), min(y0 + by + bh + pad, h)
    mask = np.zeros((my1 - my0, mx1 - mx0), np.uint8)
    cv2.fillConvexPoly(mask, hull + (x0 - mx0, y0 - my0), 1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)

    # lightly smoothed like the detection, so noise doesn't speckle the gate
    area_e = cv2.GaussianBlur(_colour(linear[my0:my1, mx0:mx1], baseline_ev), (0, 0), max(0.7, sigma))
    area_redness, area_ratio = _redness(area_e)
    mask = mask * np.clip((area_ratio - GATE_FROM) / (GATE_TO - GATE_FROM), 0.0, 1.0)
    dark = 1.0 - DARKEN * np.clip(area_redness / 0.4, 0.0, 1.0)
    # (a colour that's still pale after losing its red would stay a pale disc)
    area_lin = linear[my0:my1, mx0:mx1]
    pale = (area_lin[..., 1] + area_lin[..., 2]) / 2
    limit = PUPIL_MAX**2 / 2.0**baseline_ev
    pale = cv2.GaussianBlur(pale, (0, 0), max(0.7, sigma))
    dark = np.minimum(dark, limit / np.maximum(pale, 1e-6))
    if sigma > 1.5:  # the colour gate is steep: in a blurred eye it mustn't draw an edge of its own
        mask = cv2.GaussianBlur(mask, (0, 0), sigma / 2)
    return Spot(mx0, my0, mask, dark.astype(np.float32), float(cx + x0), float(cy + y0), float(radius))


@dataclass(frozen=True)
class _Blob:
    x: float
    y: float
    radius: float
    weight: float  # how much red there is, for picking the most promising
    sure: bool = True  # red enough and round enough by itself; False = only with a partner or a face
    of: "_Blob | None" = None  # only counts as the other eye of this one


def _blobs(linear: np.ndarray, baseline_ev: float, ox: int = 0, oy: int = 0) -> list[_Blob]:
    """The red, roundish blobs in this part of the photo (at offset ox, oy):
    sure ones and blurred ones (see _Blob). Stricter than a click: nobody
    pointed at these, so reddish skin, brows and brown irises must stay out."""
    e = cv2.GaussianBlur(_colour(linear, baseline_ev), (0, 0), 0.8)
    redness, ratio = _redness(e)
    red = ((ratio > MIN_RATIO) & (redness > MIN_REDNESS)).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(red, connectivity=8)
    found = []
    longest = max(linear.shape[:2]) / 8  # no pupil is this big
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        aspect = max(bw, bh) / min(bw, bh)
        if area < MIN_AREA or max(bw, bh) > longest or aspect > BLUR_ASPECT or area < BLUR_FILL * bw * bh:
            continue
        box = np.s_[by : by + bh, bx : bx + bw]
        inside = labels[box] == i
        colour = e[box][inside]
        how_red = float(np.median(ratio[box][inside]))
        green_blue = float(np.median(colour[:, 1] / np.maximum(colour[:, 2], 1e-4)))
        if how_red < BLUR_RATIO or green_blue > SURE_GREEN_BLUE:
            continue
        contours, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack(contours))
        solidity = area / max(cv2.contourArea(hull), 1.0)
        if solidity < BLUR_SOLIDITY:
            continue
        sure = bool(how_red >= SURE_RATIO and aspect <= MAX_ASPECT and area >= MIN_FILL * bw * bh and solidity >= MIN_SOLIDITY)
        if not sure:
            if green_blue > BLUR_GREEN_BLUE and np.percentile(redness[box][inside], 90) < BLUR_BRIGHT_RED:
                continue
            if green_blue > BLUR_GREEN_BLUE + 0.08:
                continue
            # the ring just outside the blob
            pad = int(max(bw, bh)) + 2
            y0, y1, x0, x1 = max(by - pad, 0), by + bh + pad, max(bx - pad, 0), bx + bw + pad
            part = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
            size = 2 * (max(bw, bh) // 3) + 3
            near = cv2.dilate(part, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
            far = cv2.dilate(part, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1)))
            ring = ratio[y0:y1, x0:x1][(far > 0) & (near == 0)]
            if ring.size and how_red < BLUR_CONTRAST * float(np.median(ring)):
                continue
        _, radius = cv2.minEnclosingCircle(hull)
        found.append(_Blob(float(centroids[i][0] + ox), float(centroids[i][1] + oy), max(float(radius), 1.0),
                           float(area * redness[box][inside].mean()), sure))
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
        found = find(linear, b.x, b.y, max(b.radius * 1.5, 6.0), baseline_ev, not b.sure)
        if found is not None:
            return found
    return None


class _Faces:
    """YuNet on a crop around a point, with the results back in photo pixels."""

    def __init__(self, linear: np.ndarray, baseline_ev: float):
        self.linear, self.ev = linear, baseline_ev
        self.detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (FACE_SIZE, FACE_SIZE), FACE_SCORE, 0.3, 20)
        self.checks = 0

    def whole_photo(self) -> list[np.ndarray]:
        """Eye pairs of the faces the detector sees in the photo as a whole."""
        h, w, _ = self.linear.shape
        k = min(1.0, WHOLE_SIZE / max(h, w))
        small = cv2.resize(self.linear, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1.0 else self.linear
        image = (_perceptual(small, self.ev)[..., ::-1] * 255).astype(np.uint8)
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)
        self.detector.setInputSize((FACE_SIZE, FACE_SIZE))
        return [] if faces is None else [f[4:8].reshape(2, 2) / k for f in faces]

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

    def confirm_pair(self, a: _Blob, b: _Blob, tolerance: float = EYE_TOLERANCE) -> bool:
        d = float(np.hypot(a.x - b.x, a.y - b.y))
        angle = float(np.degrees(np.arctan2(b.y - a.y, b.x - a.x)))
        for turn in (angle, angle + 180.0):  # the top of the head is on an unknown side
            for eyes in self.eyes((a.x + b.x) / 2, (a.y + b.y) / 2, d, turn):
                tol = tolerance * d
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


def _other_eye(faces: _Faces, eye: np.ndarray, d: float, partner: _Blob) -> _Blob | None:
    """Red at the spot where the face has its other eye, about as wide as the sure
    eye. A blurred pupil can be pale and in pieces, but it has to be bright red
    or blood-coloured: lashes and eyelids here are neither."""
    h, w, _ = faces.linear.shape
    reach = EYE_NEAR * d + partner.radius
    x0, y0 = max(int(eye[0] - reach), 0), max(int(eye[1] - reach), 0)
    x1, y1 = min(int(eye[0] + reach) + 1, w), min(int(eye[1] + reach) + 1, h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    e = cv2.GaussianBlur(_colour(faces.linear[y0:y1, x0:x1], faces.ev), (0, 0), 0.8)
    redness, ratio = _redness(e)
    red = (ratio > MIN_RATIO) & (redness > MIN_REDNESS)
    if red.sum() < MIN_AREA:
        return None
    colour = e[red]
    if np.median(colour[:, 1] / np.maximum(colour[:, 2], 1e-4)) > BLUR_GREEN_BLUE and np.percentile(redness[red], 90) < BLUR_BRIGHT_RED:
        return None
    points = np.argwhere(red)[:, ::-1].astype(np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(points)
    if not MIN_SIZE_MATCH <= radius / partner.radius <= 1 / MIN_SIZE_MATCH:
        return None
    return _Blob(float(cx + x0), float(cy + y0), max(float(radius), 1.0), float(red.sum()), False, partner)


def _eyes_on_faces(sure: list[_Blob], blurred: list[_Blob], faces: _Faces) -> list[_Blob]:
    """The faces first: red blobs where a face has its eyes. A sure one counts by
    itself, a blurred one with a second red eye on that face."""
    found: list[_Blob] = []

    def on(eye: np.ndarray, d: float, blobs: list[_Blob]) -> _Blob | None:
        near = [b for b in blobs if b not in found and np.hypot(b.x - eye[0], b.y - eye[1]) < EYE_NEAR * d
                and MIN_SPACING <= d / (2 * b.radius) <= MAX_SPACING]
        return max(near, key=lambda b: b.weight, default=None)

    for pair in faces.whole_photo():
        d = float(np.hypot(*(pair[0] - pair[1])))
        here = [on(eye, d, sure) or on(eye, d, blurred) for eye in pair]
        for i in (0, 1):
            if here[i] is None and here[1 - i] is not None and here[1 - i].sure:
                here[i] = _other_eye(faces, pair[i], d, here[1 - i])
        here = [b for b in here if b is not None]
        if len(here) == 2 or any(b.sure for b in here):
            found += here
    return found


def _sure_eyes(blobs: list[_Blob], faces: _Faces) -> list[_Blob]:
    """Red, round pupils in faces too small or turned for the first look: pairs
    first, then any lone ones on a face."""
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
    return eyes


def _blurred_eyes(blurred: list[_Blob], eyes: list[_Blob], faces: _Faces) -> list[_Blob]:
    """Blurred pupils: the missing partner of an eye that's sure, or both of a pair on a face."""
    found: list[_Blob] = []
    paired = set()
    for i, a in enumerate(eyes):
        for b in eyes[i + 1 :]:
            if _pair_fits(a, b, MAX_PAIR_TILT):
                paired |= {a, b}
    for a in eyes:
        if a in paired:
            continue
        fits = sorted((b for b in blurred if b not in found and _pair_fits(a, b, MAX_PAIR_TILT)), key=lambda b: -b.weight)
        for b in fits:
            if faces.checks >= MAX_CHECKS:
                break
            if faces.confirm_pair(a, b, BLUR_EYE_TOLERANCE):
                found.append(b)
                break
    rest = [b for b in blurred if b not in found]
    pairs = [(a, b) for i, a in enumerate(rest) for b in rest[i + 1 :] if _pair_fits(a, b, MAX_PAIR_TILT)]
    pairs.sort(key=lambda p: -(p[0].weight + p[1].weight))
    for a, b in pairs:
        if a in found or b in found or faces.checks >= MAX_CHECKS:
            continue
        if faces.confirm_pair(a, b):
            found += [a, b]
    return found


def _looks_right(linear: np.ndarray, spot: Spot, baseline_ev: float) -> bool:
    """A last look at a fix nobody asked for."""
    r = spot.radius
    reach = int(r * 2.2) + 2
    x0, y0 = max(int(spot.cx) - reach, 0), max(int(spot.cy) - reach, 0)
    e = cv2.GaussianBlur(_colour(linear[y0 : int(spot.cy) + reach, x0 : int(spot.cx) + reach], baseline_ev), (0, 0), 0.8)
    yy, xx = np.indices(e.shape[:2])
    d = np.hypot(xx + x0 - spot.cx, yy + y0 - spot.cy) / r
    inside, ring = d < 0.8, (d > 1.3) & (d < 2.2)
    if not inside.any() or not ring.any():
        return False
    redness, ratio = _redness(e)
    peak = float(np.percentile(redness[inside], 90))
    if peak < DIM_RED:
        return False
    green = e[..., 1]
    return not (
        np.median(ratio[inside]) < WEAK_RATIO
        and np.median(green[inside] / np.maximum(e[..., 2][inside], 1e-4)) >= WEAK_GREEN_BLUE
        and np.median(green[inside]) >= WEAK_DARKNESS * np.median(green[ring])
        and peak < WEAK_PEAK
    )


def detect(linear: np.ndarray, baseline_ev: float = 0.0) -> list[Spot]:
    """Red pupils of the whole photo."""
    blobs = sorted(_blobs(linear, baseline_ev), key=lambda b: -b.weight)
    sure = [b for b in blobs if b.sure][:MAX_BLOBS]
    blurred = [b for b in blobs if not b.sure][:MAX_BLOBS]
    if not sure and not blurred:
        return []
    faces = _Faces(linear, baseline_ev)
    eyes = _eyes_on_faces(sure, blurred, faces)
    eyes += _sure_eyes([b for b in sure if b not in eyes], faces)
    eyes += _blurred_eyes([b for b in blurred if b not in eyes], eyes, faces)
    good: dict[_Blob, Spot] = {}
    for b in sorted(eyes, key=lambda b: b.of is not None):  # others' partners last
        spot = find(linear, b.x, b.y, max(b.radius * 1.5, 6.0), baseline_ev, not b.sure)
        if spot is None:
            continue
        # A pale blurred eye passes on the strength of its partner, if that's good itself.
        if (b.of in good) if b.of is not None else _looks_right(linear, spot, baseline_ev):
            good[b] = spot
    return list(good.values())


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
