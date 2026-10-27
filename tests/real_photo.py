"""Run denoise + each sharpen model on a real photo (read-only) and save comparison crops."""
import sys, time
from pathlib import Path
import numpy as np, cv2
from freeroom import engine, imageio

src = Path(sys.argv[1]); out_dir = Path(__file__).parent / "out"; out_dir.mkdir(exist_ok=True)
photo = imageio.load(src); rgb = photo.rgb
h, w, _ = rgb.shape
den = engine.Restorer(engine.list_models("denoise")[0]); ai = rgb.copy(); den.run(rgb, ai)
# Crop with the most detail (highest Laplacian energy) in a coarse grid
lap = np.abs(cv2.Laplacian(cv2.cvtColor(ai, cv2.COLOR_RGB2GRAY), cv2.CV_32F))
cs = 400; best = max(((lap[y:y+cs, x:x+cs].mean(), y, x) for y in range(0, h-cs, cs) for x in range(0, w-cs, cs)))
_, y, x = best
cols = [rgb[y:y+cs, x:x+cs], ai[y:y+cs, x:x+cs]]
for m in engine.list_models("sharpen"):
    r = engine.Restorer(m); sh = ai.copy(); t = time.time(); r.run(ai, sh); engine.keep_detail(ai, sh)
    print(m.stem, f"{time.time()-t:.1f}s", f"{w}x{h}")
    cols.append(engine.adjust(rgb, ai, sh, .85, 1, .7, 0)[y:y+cs, x:x+cs])
grid = np.concatenate(cols, 1)
cv2.imwrite(str(out_dir / "real_compare.png"), cv2.cvtColor((np.clip(grid, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
print("crop at", x, y, "→ original | denoised | +defocus 70% | +motion 70%")
