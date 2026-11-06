"""Enhancements on the Raw tab, and AI steps stopping when switched off mid-run.

    python tests/stop_check.py

Uses a synthetic noisy photo in tests/out and its own settings (offscreen GUI).
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["XDG_CONFIG_HOME"] = str(Path(__file__).parent / "out" / "config")  # never the user's real settings
import numpy as np, cv2

out_dir = Path(__file__).parent / "out"
out_dir.mkdir(parents=True, exist_ok=True)
h, w = 3000, 4000
rng = np.random.default_rng(1)
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
img = np.stack([xx / w, yy / h, 0.5 + 0.3 * np.sin(xx / 150)], -1) + rng.normal(0, 0.06, (h, w, 3))
src = out_dir / "stop_noisy.jpg"
cv2.imwrite(str(src), (np.clip(img, 0, 1) * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 97])

from PySide6.QtWidgets import QApplication
from freeroom import app as appmod, develop, presets
presets.PRESET_DIR = out_dir / "presets_stop"
qapp = QApplication([])
win = appmod.MainWindow(); win.resize(1500, 950); win.show()
fails = []


def wait(cond=lambda: not win.busy, limit=180):
    t0 = time.time()
    while time.time() - t0 < limit:
        qapp.processEvents(); time.sleep(0.02)
        if cond():
            break
    for _ in range(5):
        qapp.processEvents()
    return time.time() - t0


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name, extra)
    if not ok:
        fails.append(name)


def started():  # the run is under way: at least one tile done
    return win.busy and win.progress.value() >= 1


win.open_path(src); wait(lambda: not win.loading)
dn_models = [win.denoise_box.itemData(i) for i in range(win.denoise_box.count())]

# 1. Noise reduction switched off while it runs: the run stops, nothing is kept
win.tabs.setCurrentWidget(win.enh_tab)
wait(started, 60)
check("denoise started", win.busy)
win.nr_group.setChecked(False)
t = wait()
check("NR off mid-run stops it", not win.busy and t < 5, f"({t:.1f}s, {win.status.text()!r})")
check("no denoise result kept", not any(k[1] == "dn" for k in win.results))

# 2. Sharpening switched off while denoise runs: denoise finishes, sharpen never runs
win.sh_group.setChecked(True); win.nr_group.setChecked(True)
wait(started, 60)
win.sh_group.setChecked(False)
wait()
kinds = sorted(k[1] for k in win.results)
check("sharpen off mid-denoise: denoise kept, no sharpen", kinds == ["dn"], str(kinds))

# 3. Another model picked mid-run: the old run stops, the new model's result is kept
if len(dn_models) > 1:
    win.photo_gen += 1; win.results.clear()  # same photo, forget results so it runs again
    win.update_pipeline()
    wait(started, 60)
    win.denoise_box.setCurrentIndex(1)
    wait()
    models = [k[2] for k in win.results if k[1] == "dn"]
    check("model change mid-run: only the new model's result", models == [dn_models[1]], str([m.stem for m in models]))
    win.denoise_box.setCurrentIndex(0); wait()

# 4. Raw tab keeps showing the enhancements
win.sliders["exposure"][0].setValue(30)
wait()
win.tabs.setCurrentIndex(0)
wait(lambda: False, 1.5)  # let the view render
v = win.view
check("Raw tab keeps the AI layer", v.dn is not None and v._small.get("dn") is not None)
sm = v._small
x0, y0, x1, y1 = sm["region"]
shown = v._render(sm["linear"], sm.get("dn"), sm.get("sharp"), False, (y0, x0, y1, x1))
plain = develop.develop(sm["linear"], v.dev_params, v.baseline_ev, v._lut)
diff = float(np.abs(shown - plain).mean())
check("Raw tab view is the enhanced photo", diff > 0.003, f"(mean diff vs development only {diff:.4f})")
before = v._render(sm["linear"], None, None, True, (y0, x0, y1, x1))
ref = develop.develop(sm["linear"], develop.DEFAULTS, v.baseline_ev, v._default_lut)
check("Raw tab 'before' is still the default development", np.allclose(before, ref))
check("no AI run started from the Raw tab", not win.busy)
win.grab().save(str(out_dir / "gui_raw_enhanced.png"))

win.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
