"""Glue between development and the AI models.

The AI models run on a fixed sRGB encoding of the *linear* photo, not on the
developed photo. Their results therefore never depend on the development
settings: every model runs at most once per photo, and the dials are applied
on top afterwards (development is per-pixel, so it can be redone instantly).

For a JPEG the encoding is exactly the original image. For a RAW it is the
photo at its base exposure, scaled down just enough that (almost) nothing
clips; the few brightest pixels that do clip fall back to the original.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import develop, engine

_SIZE = 65536
_TO_LINEAR = develop.srgb_to_linear(np.linspace(0.0, 1.0, _SIZE, dtype=np.float32))
_ENC_MAX_U = 1.0  # encode() sees x <= 1 after clipping
_TO_SRGB = np.clip(develop.linear_to_srgb(np.linspace(0.0, _ENC_MAX_U, _SIZE) ** 2), 0, 1).astype(np.float32)


def ai_scale(linear: np.ndarray, baseline_ev: float) -> float:
    """Gain from linear data to the AI's input: base exposure, pulled down so the
    brightest 0.1% still fit below white."""
    gain = 2.0**baseline_ev
    peak = float(np.percentile(linear[::4, ::4].max(axis=2), 99.9)) * gain
    return gain / max(1.0, peak)


def encode(linear: np.ndarray, g: float) -> np.ndarray:
    x = np.clip(linear * np.float32(g), 0.0, 1.0)
    np.sqrt(x, out=x)  # LUT is indexed by sqrt(linear) for fine shadow steps
    return _TO_SRGB[(x * (_SIZE - 1)).astype(np.int32)]


def decode(enc: np.ndarray, g: float, orig_linear: np.ndarray) -> np.ndarray:
    """AI output back to linear. Where the input had clipped, use the original."""
    lin = _TO_LINEAR[np.clip(enc * (_SIZE - 1) + 0.5, 0, _SIZE - 1).astype(np.int32)] / np.float32(g)
    w = np.clip((orig_linear.max(axis=2, keepdims=True) * np.float32(g) - 0.93) / 0.07, 0.0, 1.0)
    return lin + w * (orig_linear - lin)


def compose(
    linear: np.ndarray,
    dn: np.ndarray | None,
    sharp: np.ndarray | None,
    g: float,
    dev_params: dict,
    baseline_ev: float,
    enh_params: dict,
    lut: np.ndarray | None = None,
    base: np.ndarray | None = None,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Final display image: develop original + AI layers, then blend them.

    dn / sharp are AI outputs in the encoding (None = that step is off).
    base = HDR base layer for these pixels (None: `linear` is the whole photo).
    progress(label) is called before each step (see compose_steps)."""
    step = progress or (lambda label: None)
    if lut is None:
        lut = develop.build_lut(dev_params)
    if base is None and dev_params.get("hdr"):
        # from the original for all layers, so HDR treats them alike
        base = develop.hdr_base(develop.hdr_map(linear), linear.shape, linear.shape)

    def dev(x):
        return develop.develop(x, dev_params, baseline_ev, lut, base)

    step("Developing")
    orig = dev(linear)
    ai = sh = None
    if dn is not None:
        step("Developing denoised layer")
        ai = dev(decode(dn, g, linear))
    if sharp is not None:
        step("Developing sharpened layer")
        sh = dev(decode(sharp, g, linear))
    step("Blending enhancement")
    return engine.adjust(orig, orig if ai is None else ai, sh, **enh_params)


def compose_steps(dn: np.ndarray | None, sharp: np.ndarray | None) -> int:
    """How many times compose() calls progress."""
    return 2 + (dn is not None) + (sharp is not None)
