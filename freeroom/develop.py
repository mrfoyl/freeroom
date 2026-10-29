"""Simple raw development: white balance, exposure, tone, contrast, saturation, plus
HDR (local tone mapping) and black & white (colour mixer).

Works on scene-linear RGB (RAW files are loaded linear; JPEGs etc. are
linearised on load) and outputs display sRGB in [0, 1]. Every step except HDR
is per-pixel, so a screen-size proxy or a loupe crop develops exactly like the
full image. HDR works from a low-res map of the whole photo (`hdr_map`), which
a proxy or crop samples for its own pixels (`hdr_base`), so it stays exact too.

At default settings a JPEG comes out unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# exposure in EV, the rest -1..1
DEFAULTS = {
    "exposure": 0.0,
    "shadows": 0.0,
    "midtones": 0.0,
    "highlights": 0.0,
    "contrast": 0.0,
    "temperature": 0.0,
    "tint": 0.0,
    "saturation": 0.0,
    # HDR: compress the large-scale brightness range, keep/boost local detail
    "hdr": False,
    "hdr_compress": 0.5,
    "hdr_detail": 0.3,
    # Black & white: per colour, brightness overall / in highlights / in shadows
    "bw": False,
}

# Colour mixer hues (degrees) for black & white
BW_COLOURS = [("red", 0), ("orange", 30), ("yellow", 60), ("green", 120),
              ("aqua", 180), ("blue", 240), ("purple", 270), ("magenta", 300)]
BW_PARTS = ("", "_hi", "_lo")
for _name, _ in BW_COLOURS:
    for _part in BW_PARTS:
        DEFAULTS[f"bw_{_name}{_part}"] = 0.0
BW_EV = 2.0  # a colour slider at ±1 moves that colour by ±2 EV in the grey image

HDR_MAP_SIZE = 768  # long side of the low-res brightness map HDR works from
HDR_ANCHOR = float(np.log2(0.18))  # compression pivots around mid grey

# The tone LUT is indexed by sqrt(linear) so shadows get fine steps; this
# covers linear values up to LUT_MAX_U**2 = 16 (4 EV of headroom above white).
LUT_SIZE = 65536
LUT_MAX_U = 4.0


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """sRGB encoding, extended above 1 so highlight headroom survives."""
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def auto_exposure(linear: np.ndarray) -> float:
    """Baseline EV for a linear RAW: put the brightest 1% at white (like dcraw)."""
    sample = linear[::8, ::8].max(axis=2)
    p = float(np.percentile(sample, 99.0))
    if p <= 1e-6:
        return 0.0
    return float(np.clip(np.log2(1.0 / p), 0.0, 8.0))


def _gains(params: dict, baseline_ev: float) -> np.ndarray:
    t, g = params["temperature"], params["tint"]
    wb = np.array([2.0 ** (0.35 * t), 2.0 ** (-0.25 * g), 2.0 ** (-0.35 * t)], np.float32)
    wb /= wb @ LUMA  # keep brightness when shifting colour
    return wb * np.float32(2.0 ** (baseline_ev + params["exposure"]))


def _s_curve(e: np.ndarray) -> np.ndarray:
    return e * e * (3 - 2 * e)


def _s_curve_inv(e: np.ndarray) -> np.ndarray:
    return 0.5 - np.sin(np.arcsin(np.clip(1 - 2 * e, -1, 1)) / 3)


def _tone(e: np.ndarray, params: dict) -> np.ndarray:
    """Tone curve on sRGB-encoded values (may exceed 1 = headroom)."""
    s, m, h = params["shadows"], params["midtones"], params["highlights"]
    # Highlights left: a soft shoulder that compresses everything above the
    # knee, including RAW data beyond white, into the range below 1.
    knee = 1.0 - 0.6 * max(-h, 0.0)
    if knee < 1.0:
        over = e > knee
        e = np.where(over, knee + (1 - knee) * np.tanh((e - knee) / (1 - knee)), e)
    e = np.clip(e, 0.0, 1.0)
    if m:
        e = e ** (2.0 ** (-0.8 * m))
    c = params.get("contrast", 0.0)
    if c:
        # Blend towards an S-curve around the middle (or its inverse, flatter);
        # black and white stay put and the curve stays monotonic.
        curve = _s_curve(_s_curve(e)) if c > 0 else _s_curve_inv(_s_curve_inv(e))
        e = e + 0.6 * abs(c) * (curve - e)
    if s:
        e = e + 0.35 * s * 4 * e * (1 - e) ** 3
    if h > 0:
        e = e + 0.35 * h * 4 * e**3 * (1 - e)
    return np.clip(e, 0.0, 1.0)


def _guided(i: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Self-guided filter: smooths within regions, keeps strong edges."""
    def box(a):
        return cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)

    m = box(i)
    var = np.maximum(box(i * i) - m * m, 0)
    a = var / (var + eps)
    return box(a) * i + box(m - a * m)


def hdr_map(linear: np.ndarray) -> np.ndarray:
    """Edge-preserving smooth log2 luminance of the whole photo, at low res."""
    h, w = linear.shape[:2]
    s = HDR_MAP_SIZE / max(h, w)
    if s < 1:
        linear = cv2.resize(linear, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    lum = np.log2(np.maximum(linear @ LUMA, 1e-6)).astype(np.float32)
    return _guided(lum, max(2, round(max(lum.shape) / 24)), 0.25)


def hdr_base(hmap: np.ndarray, full_shape: tuple, out_shape: tuple, region: tuple | None = None) -> np.ndarray:
    """The HDR base layer for `region` (y0, x0, y1, x1; default the whole photo
    of `full_shape`) at `out_shape`, sampled from `hdr_map`."""
    H, W = full_shape[:2]
    oh, ow = out_shape[:2]
    mh, mw = hmap.shape
    y0, x0, y1, x1 = region or (0, 0, H, W)
    sx, sy = (x1 - x0) / ow * mw / W, (y1 - y0) / oh * mh / H
    # output pixel centre -> map pixel coordinate
    m = np.float32([[sx, 0, x0 * mw / W + 0.5 * sx - 0.5], [0, sy, y0 * mh / H + 0.5 * sy - 0.5]])
    return cv2.warpAffine(hmap, m, (ow, oh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def _hdr(x: np.ndarray, params: dict, base: np.ndarray) -> np.ndarray:
    """Local tone mapping in log2 luminance: pull the base layer towards mid
    grey (shadows up, highlights down), scale the detail on top. Hue is kept."""
    lum = np.log2(np.maximum(x @ LUMA, 1e-6))
    c, d = 0.85 * params["hdr_compress"], params["hdr_detail"]
    shift = -c * (base - HDR_ANCHOR) + d * (lum - base)
    return x * np.exp2(np.clip(shift, -8, 8)).astype(np.float32)[..., None]


def _smoothstep(a: float, b: float, t: np.ndarray) -> np.ndarray:
    t = np.clip((t - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _mono(x: np.ndarray, params: dict) -> np.ndarray:
    """Black & white: luminance, with each colour made brighter/darker by its
    slider, plus its High / Low slider in the bright / dark parts of the photo."""
    y = x @ LUMA
    e = np.sqrt(np.maximum(x, 0))  # roughly perceptual, for hue and colourfulness
    mx, mn = e.max(-1), e.min(-1)
    d = np.maximum(mx - mn, 1e-9)
    r, g, b = e[..., 0], e[..., 1], e[..., 2]
    hue = 60 * np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
    chroma = (mx - mn) / np.maximum(mx, 1e-6)  # 0 for greys: they stay as they are
    hues = [h for _, h in BW_COLOURS] + [360]

    def mix(part):
        vals = [params.get(f"bw_{n}{part}", 0.0) for n, _ in BW_COLOURS]
        return np.interp(hue, hues, vals + vals[:1]) if any(vals) else None

    ev = np.zeros_like(y)
    hi_w = _smoothstep(0.15, 0.85, np.sqrt(np.clip(y, 0, 1)))
    for part, w in (("", 1.0), ("_hi", hi_w), ("_lo", 1 - hi_w)):
        m = mix(part)
        if m is not None:
            ev += m * w
    grey = (y * np.exp2(BW_EV * ev * chroma)).astype(np.float32)
    return np.repeat(grey[..., None], 3, axis=-1)


def build_lut(params: dict) -> np.ndarray:
    u = np.linspace(0.0, LUT_MAX_U, LUT_SIZE, dtype=np.float64)
    return _tone(linear_to_srgb(u * u), params).astype(np.float32)


def develop(
    linear: np.ndarray,
    params: dict,
    baseline_ev: float = 0.0,
    lut: np.ndarray | None = None,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """Linear RGB -> developed display sRGB (float32, 0..1).

    `base` = HDR base layer for these pixels (`hdr_base`); None means `linear`
    is the whole photo and it's computed here."""
    if lut is None:
        lut = build_lut(params)
    x = linear * _gains(params, baseline_ev)
    np.maximum(x, 0.0, out=x)
    if params.get("hdr"):
        if base is None:
            base = hdr_base(hdr_map(linear), linear.shape, linear.shape)
        x = _hdr(x, params, base + np.float32(baseline_ev + params["exposure"]))
    if params.get("bw"):
        x = _mono(x, params)
    np.sqrt(x, out=x)
    idx = np.minimum(x * ((LUT_SIZE - 1) / LUT_MAX_U), LUT_SIZE - 1).astype(np.int32)
    out = lut[idx]
    sat = params["saturation"]
    if sat:
        y = (out @ LUMA)[..., None]
        out = y + (out - y) * np.float32(1.0 + sat)
        np.clip(out, 0.0, 1.0, out=out)
    return out


def beyond_white(linear: np.ndarray, params: dict, baseline_ev: float = 0.0) -> float:
    """Fraction of pixels with a channel past white before tone mapping, i.e.
    highlight data that is clipped at Highlights 0 and can be pulled back."""
    x = linear * _gains(params, baseline_ev)
    return float((x.max(axis=2) > 1.0).mean())


def histogram(rgb: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel histograms (3 x bins) + fraction clipped dark / bright per channel."""
    idx = np.clip((rgb * (bins - 1) + 0.5).astype(np.int32), 0, bins - 1)
    hist = np.stack([np.bincount(idx[..., c].ravel(), minlength=bins) for c in range(3)])
    n = idx.shape[0] * idx.shape[1]
    return hist, hist[:, 0] / n, hist[:, -1] / n
