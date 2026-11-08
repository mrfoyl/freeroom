"""AI selection for layers: SAM 2.1 small (Meta, Apache-2.0) through transformers.

The photo is read once (image embedding, about half a second on a GPU); each
click or brush stroke then only runs the small mask decoder. Masks come out at
the "work" resolution (longest side WORK px), with soft edges that a guided
filter pulls onto the photo's own edges.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "segment" / "sam2.1-hiera-small"
WORK = 2560  # longest side of masks and of the image the AI sees


def available() -> bool:
    return (MODEL_DIR / "model.safetensors").exists()


def work_shape(h: int, w: int) -> tuple[int, int]:
    k = min(1.0, WORK / max(h, w))
    return max(1, round(h * k)), max(1, round(w * k))


def guided(guide: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter (He et al.): smooths p while following the edges of guide."""
    k = (2 * r + 1, 2 * r + 1)

    def mean(x):
        return cv2.boxFilter(x, -1, k)

    mi, mp = mean(guide), mean(p)
    a = (mean(guide * p) - mi * mp) / (mean(guide * guide) - mi * mi + eps)
    b = mp - a * mi
    return mean(a) * guide + mean(b)


class Segmenter:
    """One photo at a time: set_image(), then predict() per prompt set."""

    def __init__(self):
        self.model = self.processor = None
        self.device = None
        self.emb = None
        self.size = (0, 0)
        self.guide: np.ndarray | None = None

    def _load(self, device=None):
        import torch
        from transformers import Sam2Model, Sam2Processor

        from .engine import _device

        self.device = device or _device()
        self.processor = Sam2Processor.from_pretrained(MODEL_DIR)
        self.model = Sam2Model.from_pretrained(MODEL_DIR).to(self.device).eval()
        self.torch = torch

    def set_image(self, rgb: np.ndarray):
        """rgb: HxWx3 display-encoded float 0..1 or uint8, at work resolution."""
        if self.model is None:
            self._load()
        img8 = rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
        self.size = img8.shape[:2]
        self.guide = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        try:
            self._embed(img8)
        except Exception:
            if self.device.type == "cpu":
                raise
            self._load(self.torch.device("cpu"))  # e.g. an op Metal lacks
            self._embed(img8)

    def _embed(self, img8: np.ndarray):
        inputs = self.processor(images=img8, return_tensors="pt")
        with self.torch.inference_mode():
            self.emb = self.model.get_image_embeddings(inputs["pixel_values"].to(self.device))

    def predict(self, points: list[tuple[float, float]], labels: list[int]) -> np.ndarray:
        """Probability mask (float32, work size) for points in work pixels;
        label 1 = part of it, 0 = not."""
        torch = self.torch
        h, w = self.size
        single = len(points) == 1
        p = self.processor(input_points=[[[list(map(float, pt)) for pt in points]]],
                           input_labels=[[list(map(int, labels))]], original_sizes=[[h, w]], return_tensors="pt")
        with torch.inference_mode():
            out = self.model(input_points=p["input_points"].to(self.device), input_labels=p["input_labels"].to(self.device),
                             image_embeddings=self.emb, multimask_output=single)
        masks = out.pred_masks.float().cpu()
        if single:  # one click is ambiguous (part / object / whole): take the surest
            best = int(out.iou_scores[0, 0].float().argmax())
            masks = masks[:, :, best : best + 1]
        logits = self.processor.post_process_masks(masks, p["original_sizes"], binarize=False)[0][0, 0].numpy()
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        return self.refine(prob.astype(np.float32))

    def refine(self, prob: np.ndarray) -> np.ndarray:
        """Pull a soft mask onto the photo's edges."""
        r = max(4, round(max(self.size) / 320))
        return np.clip(guided(self.guide, prob, r, 1e-3), 0, 1).astype(np.float32)
