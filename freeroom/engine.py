"""AI denoising + AI sharpening on the GPU (tiled), and the cheap slider
adjustments on top. GPU = NVIDIA (CUDA) or Apple Silicon (Metal), else CPU.

Each model runs once per photo: denoise first, then the optional deblur model
on the denoised result (deblurring a noisy image just amplifies the noise).
The sliders only blend the model outputs, so moving them is instant.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

# Apple GPU: run the few ops Metal lacks on the CPU instead of failing (must be set before torch loads)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_EXTS = {".pth", ".pt", ".safetensors", ".ckpt"}

# Rec.709 luma weights
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def list_models(kind: str) -> list[Path]:
    """Models in models/<kind>/ ("denoise" or "sharpen")."""
    folder = MODELS_DIR / kind
    if not folder.is_dir():
        return []
    models = [p for p in folder.iterdir() if p.suffix.lower() in MODEL_EXTS]
    # Safest defaults first: PSNR SCUNet (no invented texture), defocus deblur.
    models.sort(key=lambda p: (0 if "real_psnr" in p.name or "defocus" in p.name else 1, p.name))
    return models


def pretty_name(path: Path) -> str:
    """Short menu label, e.g. scunet_color_real_psnr -> "SCUNet psnr"."""
    names = {
        "scunet_color_real_psnr": "SCUNet psnr (faithful)",
        "scunet_color_real_gan": "SCUNet gan (crisper)",
        "restormer_single_image_defocus_deblurring": "Defocus (soft / out of focus)",
        "restormer_motion_deblurring": "Motion (camera shake)",
    }
    return names.get(path.stem, path.stem)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name() -> str:
    dev = _device()
    if dev.type == "cuda":
        return torch.cuda.get_device_name(0)
    if dev.type == "mps":
        return "Apple GPU (Metal)"
    return "CPU (no GPU found, will be slow)"


class Restorer:
    def __init__(self, model_path: Path, tile: int = 512, overlap: int = 48):
        import spandrel_extra_arches  # Restormer lives here
        from spandrel import ModelLoader

        spandrel_extra_arches.install(ignore_duplicates=True)

        self.device = _device()
        desc = ModelLoader().load_from_file(str(model_path))
        if desc.input_channels != 3 or desc.output_channels != 3 or desc.scale != 1:
            raise ValueError(
                f"{model_path.name} is not a 1x RGB restoration model "
                f"(in={desc.input_channels}, out={desc.output_channels}, scale={desc.scale})"
            )
        self.desc = desc.to(self.device).eval()
        self.dtype = torch.float32  # also on Metal: half precision there is less proven
        if self.device.type == "cuda" and desc.supports_bfloat16:
            self.dtype = torch.bfloat16
        elif self.device.type == "cuda" and desc.supports_half:
            self.dtype = torch.float16
        self.desc.model.to(self.dtype)
        self.tile = tile
        self.overlap = overlap
        self.name = model_path.stem

    @torch.inference_mode()
    def run(
        self,
        rgb: np.ndarray,
        out: np.ndarray,
        progress: Callable[[int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        """Run the model on `rgb` into `out` (same shape), tile by tile.

        `out` is updated as each tile finishes so a preview can show partial
        results. Returns False if cancelled.
        """
        h, w, _ = rgb.shape
        t, o = self.tile, self.overlap
        stride = t - o
        # Reflect-pad so edge tiles see real context and every tile is full size.
        ph, pw = max(h + 2 * o, t), max(w + 2 * o, t)
        padded = np.pad(rgb, ((o, ph - h - o), (o, pw - w - o), (0, 0)), mode="reflect")

        ys = _starts(ph, t, stride)
        xs = _starts(pw, t, stride)
        window = _feather(t, o)
        acc = np.zeros((ph, pw, 3), np.float32)
        wsum = np.zeros((ph, pw, 1), np.float32)

        total, done = len(ys) * len(xs), 0
        for y in ys:
            for x in xs:
                if cancelled and cancelled():
                    return False
                tile = padded[y : y + t, x : x + t]
                inp = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1)))[None]
                res = self.desc(inp.to(self.device, self.dtype))
                res = res[0].float().clamp_(0, 1).permute(1, 2, 0).cpu().numpy()

                acc[y : y + t, x : x + t] += res * window
                wsum[y : y + t, x : x + t] += window

                # Publish the finished part of this tile into the visible output.
                y0, y1 = max(y, o), min(y + t, o + h)
                x0, x1 = max(x, o), min(x + t, o + w)
                if y1 > y0 and x1 > x0:
                    out[y0 - o : y1 - o, x0 - o : x1 - o] = (
                        acc[y0:y1, x0:x1] / np.maximum(wsum[y0:y1, x0:x1], 1e-6)
                    )
                done += 1
                if progress:
                    progress(done, total)

        out[:] = acc[o : o + h, o : o + w] / np.maximum(wsum[o : o + h, o : o + w], 1e-6)
        return True

    def close(self):
        del self.desc
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()


def keep_detail(base: np.ndarray, sharp: np.ndarray, sigma: float = 6.0) -> None:
    """Keep only the fine detail the sharpen model added (in place).

    Deblur models also nudge overall tone/colour a little; high-passing their
    change keeps the sharpening but leaves the denoised colours alone.
    """
    delta = sharp - base
    delta -= cv2.GaussianBlur(delta, (0, 0), sigma)
    np.add(base, delta, out=sharp)


def _starts(size: int, tile: int, stride: int) -> list[int]:
    starts = list(range(0, max(size - tile, 0), stride))
    starts.append(size - tile)
    return starts


def _feather(tile: int, overlap: int) -> np.ndarray:
    ramp = np.ones(tile, np.float32)
    if overlap > 0:
        r = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
        ramp[:overlap] = r
        ramp[-overlap:] = r[::-1]
    return (ramp[:, None] * ramp[None, :])[..., None]


def adjust(
    orig: np.ndarray,
    ai: np.ndarray,
    sharp: np.ndarray | None,
    luminance: float,
    color: float,
    ai_sharpen: float,
    sharpen: float,
) -> np.ndarray:
    """Blend the AI results with the original.

    luminance / color: 0..1, how much of the brightness / colour noise to remove.
    ai_sharpen: 0..1.5, how much of the deblur model's added detail to apply
        (`sharp` is the deblurred version of `ai`; >1 exaggerates it).
    sharpen: 0..1, classic unsharp mask on brightness.
    """
    diff = orig - ai
    diff_y = diff @ LUMA
    diff_c = diff - diff_y[..., None]
    out = ai + (1.0 - luminance) * diff_y[..., None] + (1.0 - color) * diff_c
    if sharp is not None and ai_sharpen > 0:
        out += ai_sharpen * (sharp - ai)
    if sharpen > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 1.0)
        detail = (out - blur) @ LUMA
        out = out + (1.5 * sharpen) * detail[..., None]
    return np.clip(out, 0.0, 1.0, out=out)
