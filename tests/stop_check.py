"""Ticking a section starts its AI step, unticking stops it mid-run; the Cancel
button unticks; enhancements show on the Raw tab; saving while a step runs.

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

# 0. Sections start unticked and nothing runs
win.tabs.setCurrentWidget(win.enh_tab)
wait(lambda: False, 1)
check("sections unticked by default, nothing runs",
      not win.nr_group.isChecked() and not win.sh_group.isChecked() and not win.busy and not win.results)

# 1. Ticking noise reduction starts it; unticking mid-run stops it, nothing is kept
win.nr_group.setChecked(True)
check("ticking NR starts it", win.busy)
wait(started, 60)
win.nr_group.setChecked(False)
t = wait()
check("NR unticked mid-run stops it", not win.busy and t < 1, f"({t:.2f}s, {win.status.text()!r})")
check("no denoise result kept", not any(k[1] == "dn" for k in win.results))

# 1b. Sharpening (the slower model) unticked mid-run stops just as fast
win.sh_group.setChecked(True)
wait(started, 60)
win.sh_group.setChecked(False)
t = wait()
check("sharpen unticked mid-run stops it", not win.busy and t < 1 and not win.results, f"({t:.2f}s)")

# 1c. The Cancel button unticks the running section
win.nr_group.setChecked(True)
wait(started, 60)
win.stop_enhancement()
t = wait()
check("Cancel unticks and stops", not win.nr_group.isChecked() and not win.busy and t < 1 and not win.results,
      f"({t:.2f}s)")

# 2. Sharpening unticked while denoise runs: denoise finishes, sharpen never runs
win.nr_group.setChecked(True); win.sh_group.setChecked(True)
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
    win.denoise_box.setCurrentIndex(1)  # runs when the old one stops
    wait(); wait()
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
check("nothing left running", not win.busy)
win.grab().save(str(out_dir / "gui_raw_enhanced.png"))

# 5. Saving while an AI step runs asks: save now (the step goes on) or wait, then save
from unittest import mock
from PySide6.QtWidgets import QMessageBox


def save_while_running(choice: str, name: str):
    win.photo_gen += 1; win.results.clear()  # forget results so it runs again
    win.update_pipeline()
    wait(started, 60)
    f = out_dir / name; f.unlink(missing_ok=True)
    asked = []

    def answer(box):
        asked.append(box.text())
        box.clickedButton = lambda: next(b for b in box.buttons() if b.text() == choice)
    with mock.patch.object(QMessageBox, "exec", answer), \
         mock.patch.object(appmod.QFileDialog, "getSaveFileName", return_value=(str(f), "")):
        win.save_dialog()
    return f, asked


f, asked = save_while_running("Save now", "stop_saved_now.jpg")
check("save while running asks", bool(asked), str(asked))
wait(lambda: f.exists() and not win.saving, 120)
check("Save now: saved while the step goes on", f.exists() and win.nr_group.isChecked(), win.status.text())
wait()
check("Save now: the step still finished", any(k[1] == "dn" for k in win.results))
f, asked = save_while_running("Wait, then save", "stop_saved_after.jpg")
check("Wait: not saved while it runs", win.busy and not f.exists())
wait(lambda: f.exists() and not win.saving, 300)
check("Wait: saved once the enhancement is done", f.exists() and any(k[1] == "dn" for k in win.results),
      win.status.text())

win._leave_photo = lambda: True  # quitting asks to keep the edits
win.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
