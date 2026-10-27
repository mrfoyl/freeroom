"""Simple raw development: white balance, exposure, tone, saturation.

Works on scene-linear RGB (RAW files are loaded linear; JPEGs etc. are
linearised on load) and outputs display sRGB in [0, 1]. Every step is
per-pixel, so a screen-size proxy or a loupe crop develops exactly like the
full image, and the full image only has to be developed for enhancement/saving.

At default settings a JPEG comes out unchanged.
"""

from __future__ import annotations

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# exposure in EV, the rest -1..1
DEFAULTS = {
    "exposure": 0.0,
    "shadows": 0.0,
    "midtones": 0.0,
    "highlights": 0.0,
    "temperature": 0.0,
    "tint": 0.0,
    "saturation": 0.0,
}

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
    if s:
        e = e + 0.35 * s * 4 * e * (1 - e) ** 3
    if h > 0:
        e = e + 0.35 * h * 4 * e**3 * (1 - e)
    return np.clip(e, 0.0, 1.0)


def build_lut(params: dict) -> np.ndarray:
    u = np.linspace(0.0, LUT_MAX_U, LUT_SIZE, dtype=np.float64)
    return _tone(linear_to_srgb(u * u), params).astype(np.float32)


def develop(linear: np.ndarray, params: dict, baseline_ev: float = 0.0, lut: np.ndarray | None = None) -> np.ndarray:
    """Linear RGB -> developed display sRGB (float32, 0..1)."""
    if lut is None:
        lut = build_lut(params)
    x = linear * _gains(params, baseline_ev)
    np.maximum(x, 0.0, out=x)
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
