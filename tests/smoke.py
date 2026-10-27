"""Headless smoke test: develop + engine on a synthetic noisy photo, optional
RAW file (read-only), presets, and the GUI offscreen.

    python tests/smoke.py [some.CR2]
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np, cv2

out_dir = Path(__file__).parent / "out"
out_dir.mkdir(parents=True, exist_ok=True)
raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

# Synthetic "photo": smooth gradients + edges + text, then realistic-ish noise
h, w = 3000, 4000
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
clean = np.stack([xx / w, yy / h, 0.5 + 0.3 * np.sin(xx / 150) * np.cos(yy / 200)], -1)
cv2.circle(clean, (2000, 1500), 600, (0.9, 0.2, 0.1), -1)
m = np.zeros((h, w), np.uint8); cv2.putText(m, "DENOISE 123", (300, 2600), cv2.FONT_HERSHEY_SIMPLEX, 8, 255, 20); clean[m > 0] = 1.0
rng = np.random.default_rng(0)
lum = rng.normal(0, 0.06, (h, w, 1)) * np.sqrt(clean.mean(-1, keepdims=True) + 0.05)
chroma = cv2.GaussianBlur(rng.normal(0, 0.05, (h, w, 3)).astype(np.float32), (0, 0), 2)
noisy = np.clip(clean + lum + chroma, 0, 1).astype(np.float32)
src = out_dir / "noisy.jpg"
cv2.imwrite(str(src), cv2.cvtColor((noisy * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 97])

from freeroom import develop, engine, imageio, presets
def psnr(a, ref=clean): return 10 * np.log10(1 / np.mean((a - ref) ** 2))

# --- development
photo = imageio.load(src)
jpeg8 = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
t0 = time.time(); dev0 = develop.develop(photo.linear, develop.DEFAULTS)
print(f"develop 12MP: {time.time()-t0:.2f}s   JPEG identity max err: {np.abs(dev0 - jpeg8).max() * 255:.2f}/255")
p = dict(develop.DEFAULTS)
for k, v in [("exposure", 1.0), ("shadows", 1), ("midtones", 1), ("highlights", 1), ("highlights", -1),
             ("temperature", 1), ("tint", 1), ("saturation", 1), ("saturation", -1)]:
    q = dict(p, **{k: v}); d = develop.develop(photo.linear, q)
    print(f"  {k:12s}{v:+} mean RGB {d.mean(axis=(0, 1)).round(3)}  (default {dev0.mean(axis=(0, 1)).round(3)})")
hist, lo, hi = develop.histogram(dev0)
print("histogram sum ok:", (hist.sum(axis=1) == dev0.shape[0] * dev0.shape[1]).all())

# --- engine on the developed image
for mp in engine.list_models("denoise"):
    d = engine.Restorer(mp); ai = dev0.copy(); t0 = time.time(); d.run(dev0, ai)
    print(f"{mp.stem}: {time.time()-t0:.1f}s PSNR noisy={psnr(dev0):.2f} ai={psnr(ai):.2f} nan={np.isnan(ai).any()}")

# --- RAW (read-only)
if raw_path:
    t0 = time.time(); rp = imageio.load(raw_path)
    print(f"RAW {raw_path.name}: {rp.linear.shape[1]}x{rp.linear.shape[0]} load {time.time()-t0:.1f}s baseline +{rp.baseline_ev:.2f} EV")
    t0 = time.time(); rd = develop.develop(rp.linear, develop.DEFAULTS, rp.baseline_ev)
    print(f"  develop full: {time.time()-t0:.2f}s  beyond white: {develop.beyond_white(rp.linear[::4, ::4], develop.DEFAULTS, rp.baseline_ev) * 100:.2f}%")
    small = cv2.resize(rp.linear, (1200, int(1200 * rp.linear.shape[0] / rp.linear.shape[1])), interpolation=cv2.INTER_AREA)
    row = [develop.develop(small, dict(develop.DEFAULTS, **kw), rp.baseline_ev) for kw in
           ({}, {"highlights": -0.8}, {"exposure": 0.7, "shadows": 0.6, "highlights": -0.6, "saturation": 0.25})]
    cv2.imwrite(str(out_dir / "raw_dev.jpg"), cv2.cvtColor((np.concatenate(row, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# --- presets (into a scratch dir, not the user's config)
presets.PRESET_DIR = out_dir / "presets"
import shutil; shutil.rmtree(presets.PRESET_DIR, ignore_errors=True)  # else "overwrite?" dialog blocks

# --- GUI
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from freeroom import app as appmod
qapp = QApplication([])
win = appmod.MainWindow(); win.resize(1500, 950); win.show()

def wait(cond=lambda: not win.busy, limit=180):
    t0 = time.time()
    while time.time() - t0 < limit:
        qapp.processEvents(); time.sleep(0.03)
        if cond(): break
    for _ in range(5): qapp.processEvents()
    return f"{time.time()-t0:.1f}s"

win.open_path(raw_path or src)
win.sliders["exposure"][0].setValue(30); win.sliders["shadows"][0].setValue(40); win.sliders["highlights"][0].setValue(-50)
print("dev tab, AI not run:", win.ai is None and not win.busy)
win.view.mouse = QPointF(win.view.width() * 0.45, win.view.height() * 0.5); win.view.update(); qapp.processEvents()
win.grab().save(str(out_dir / "gui_develop.png"))

win.tabs.setCurrentWidget(win.enh_tab); print("enhance tab:", wait(), win.status.text())
win.sh_group.setChecked(True); print("+sharpen:", wait())
win.grab().save(str(out_dir / "gui_enhance.png"))
win.tabs.setCurrentIndex(0); win.tabs.setCurrentWidget(win.enh_tab); print("back to enhance, no change:", wait(), "busy:", win.busy)
win.tabs.setCurrentIndex(0); win.sliders["midtones"][0].setValue(20); win.tabs.setCurrentWidget(win.enh_tab)
print("dev changed -> re-enhance:", win.busy, wait())

# presets
win.preset_box.setCurrentIndex(0)
from unittest import mock
with mock.patch.object(appmod.QInputDialog, "getText", return_value=("Test look", True)):
    win.save_preset()
print("preset saved:", presets.names(), (presets.PRESET_DIR / "Test look.json").read_text()[:60].replace("\n", " "))
saved = win.settings()
win._reset(appmod.DEV_KEYS + appmod.ENH_KEYS); win.sh_group.setChecked(False); wait()
win.preset_chosen(win.preset_box.findData("Test look")); wait()
print("preset round trip identical:", win.settings() == saved)

# save from the development tab after a change: must develop + enhance first
win.tabs.setCurrentIndex(0); win.sliders["saturation"][0].setValue(15)
out_file = out_dir / "saved_from_dev_tab.jpg"; out_file.unlink(missing_ok=True)
with mock.patch.object(appmod.QFileDialog, "getSaveFileName", return_value=(str(out_file), "")):
    win.save_dialog()
print("save from dev tab:", wait(lambda: out_file.exists()), out_file.exists(), win.status.text())
win.close()
