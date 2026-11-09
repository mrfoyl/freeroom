"""Layers with AI selection: magic select, foreground/background brush, settings
per layer, AI steps per layer, saving with layers, undo, invert/feather/opacity.

    python tests/layers_check.py photo x y   (photo only read; x, y = a point on an object)

Saved files go to tests/out and are deleted afterwards.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["XDG_CONFIG_HOME"] = str(Path(__file__).parent / "out" / "config")  # never the user's real settings
from unittest import mock

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from freeroom import app as appmod, develop, presets

out_dir = Path(__file__).parent / "out"
presets.PRESET_DIR = out_dir / "presets_layers"
photo, px, py = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
qapp = QApplication([])
win = appmod.MainWindow(); win.resize(1500, 950); win.show()
fails = []


def wait(cond=lambda: not win.busy, limit=120):
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


def render(stack=True):
    v = win.view
    sm = v._small
    x0, y0, x1, y1 = sm["region"]
    return v._render(sm["linear"], sm.get("dn"), sm.get("sharp"), False, (y0, x0, y1, x1),
                     lambda a: sm.get(f"ai{id(a)}"), None if stack else [])


def at(img, x, y):  # view pixel for photo point
    v = win.view
    x0, y0, x1, y1 = v._small["region"]
    h, w = img.shape[:2]
    return img[int((y - y0) / (y1 - y0) * h), int((x - x0) / (x1 - x0) * w)]


win.open_path(photo); wait(lambda: not win.loading)
H, W = win.photo.linear.shape[:2]
check("no layers, Base active", win.active is None and not win.layers and win.layer_list.count() == 1)
check("Layers panel folded at start", not win.layers_toggle.isChecked() and not win.layers_body.isVisible())
win.grab().save(str(out_dir / "layers_folded.png"))
win.layers_toggle.click(); wait(lambda: False, 0.2)
check("header click opens it", win.layers_body.isVisible())

# 1. New layer: magic select on, the AI reads the photo, a click selects the object
win.add_layer()
check("new layer is active, magic select on", win.active == 0 and win.view.mask_tool == "magic")
t = wait(lambda: win.segworker.gen == win.photo_gen, 60)
check("AI selection read the photo", win.segworker.gen == win.photo_gen, f"({t:.1f}s)")
L = win.layers[0]
win.view.mask_clicked.emit(px, py, True)
t = wait(lambda: L.ai is not None, 30)
frac = float(L.mask.mean())
k = L.scale
check("magic click selects the object", L.ai is not None and L.mask[int(py * k), int(px * k)] > 0.9 and 0.002 < frac < 0.6,
      f"({t:.2f}s, {frac * 100:.1f}% of the photo)")
win.grab().save(str(out_dir / "layers_selection.png"))
check("header names the layer", "Layer 1" in win.layers_toggle.text(), repr(win.layers_toggle.text()))

# 2. The tabs now edit the layer: exposure -1 EV only inside the selection
base_before = render(stack=False)
win.sliders["exposure"][0].setValue(-100)
wait(lambda: False, 0.5)
img = render()
inside = float(np.mean(at(base_before, px, py))) - float(np.mean(at(img, px, py)))
far = (W * 0.03, H * 0.97)  # a corner, away from the object
outside = abs(float(np.mean(at(img, *far))) - float(np.mean(at(base_before, *far))))
check("layer exposure darkens inside only", inside > 0.05 and outside < 0.01, f"(inside -{inside:.3f}, outside {outside:.4f})")
check("Base settings unchanged", win.settings()["development"]["exposure"] == 0.0)

# 3. Back to Base: the tabs show the Base, the layer keeps its edit
win.layer_list.setCurrentRow(0); wait(lambda: False, 0.3)
check("Base active, tabs show Base exposure", win.active is None and win.sliders["exposure"][0].value() == 0)
check("layer kept its exposure", L.settings["development"]["exposure"] == -1.0)
win.layer_list.setCurrentRow(1); wait(lambda: False, 0.3)
check("layer again, tabs show its exposure", win.active == 0 and win.sliders["exposure"][0].value() == -100)

# 4. Brush: paint foreground on the clothes below (grows), background over the click (removes)
win._set_mask_tool("brush")
before = float(L.mask.mean())
stroke = [(px, py + H * 0.25 + i * H * 0.01) for i in range(10)]
win.view.mask_stroked.emit(stroke, H / 60, True)
wait(lambda: False, 1.5)
check("foreground stroke grows the selection", float(L.mask.mean()) > before, f"({before * 100:.1f}% -> {L.mask.mean() * 100:.1f}%)")
check("painted pixels are in", L.mask[int(stroke[5][1] * k), int(stroke[5][0] * k)] > 0.99)
win.view.mask_stroked.emit([(px - 5, py), (px + 5, py)], H / 80, False)
wait(lambda: False, 1.5)
check("background stroke removes pixels", L.mask[int(py * k), int(px * k)] < 0.01)
n_undo = len(L.undo)
win.mask_undo(); win.mask_undo()
check("undo twice goes back to the magic click", len(L.undo) == n_undo - 2 and L.mask[int(py * k), int(px * k)] > 0.9)

# 4b. Ctrl subtracts: Ctrl+click (magic) and a Ctrl brush stroke, with real mouse events
from PySide6.QtTest import QTest
CTRL, LEFT = appmod.Qt.KeyboardModifier.ControlModifier, appmod.Qt.MouseButton.LeftButton
pos = win.view._to_widget(px, py).toPoint()
win._set_mask_tool("magic"); wait(lambda: False, 0.2)
jx, jy = px, py + H * 0.3  # the jacket
QTest.mouseClick(win.view, LEFT, appmod.Qt.KeyboardModifier.NoModifier, win.view._to_widget(jx, jy).toPoint())
wait(lambda: L.mask[int(jy * k), int(jx * k)] > 0.5, 15); wait(lambda: False, 0.5)
grown = float(L.mask.mean())
QTest.mouseClick(win.view, LEFT, CTRL, win.view._to_widget(jx, jy + H * 0.05).toPoint())
wait(lambda: float(L.mask.mean()) < grown * 0.8, 15)
check("Ctrl+click with magic select subtracts", L.points[-1][2] == 0 and float(L.mask.mean()) < grown * 0.8
      and L.mask[int(py * k), int(px * k)] > 0.5,
      f"(face + jacket {grown * 100:.1f}% -> {L.mask.mean() * 100:.1f}% after Ctrl+click on the jacket)")
win.mask_undo(); win.mask_undo()
win._set_mask_tool("brush"); wait(lambda: False, 0.2)
n = len(L.undo)
a, b = win.view._to_widget(px - 5, py).toPoint(), win.view._to_widget(px + 5, py).toPoint()
QTest.mousePress(win.view, LEFT, CTRL, a)
QTest.mouseMove(win.view, b)
QTest.mouseRelease(win.view, LEFT, CTRL, b)
wait(lambda: len(L.undo) > n, 15); wait(lambda: False, 1.0)
check("Ctrl brush stroke subtracts", L.mask[int(py * k), int(px * k)] < 0.01 and L.paint[int(py * k), int(px * k)] < 0)
win.mask_undo()
check("undo brings it back", L.mask[int(py * k), int(px * k)] > 0.9)
win._set_mask_tool(None)

# 4c. Hovering a layer in the list shows its area, leaving shows the active one again
win.show_mask_box.setChecked(False)
win.layer_list.setCurrentRow(0); wait(lambda: False, 0.3)
check("Base active: no selection shown", win.view.mask_show is None)
vp = win.layer_list.viewport()
QTest.mouseMove(vp, win.layer_list.visualItemRect(win.layer_list.item(1)).center()); wait(lambda: False, 0.2)
check("hover a layer shows its area", win.view.mask_show is L.mask)
QTest.mouseMove(vp, win.layer_list.visualItemRect(win.layer_list.item(0)).center()); wait(lambda: False, 0.2)
check("hover Base shows nothing", win.view.mask_show is None)
QTest.mouseMove(vp, win.layer_list.visualItemRect(win.layer_list.item(1)).center()); wait(lambda: False, 0.2)
qapp.sendEvent(vp, appmod.QEvent(appmod.QEvent.Type.Leave)); wait(lambda: False, 0.2)
check("leaving the list hides it again", win.view.mask_show is None)
win.layer_list.setCurrentRow(1); wait(lambda: False, 0.3)

# 5. Invert, feather, opacity, hide
win.invert_box.setChecked(True); wait(lambda: False, 0.2)
check("invert", L.mask[int(py * k), int(px * k)] < 0.1 and L.mask[int(far[1] * k), int(far[0] * k)] > 0.9)
win.invert_box.setChecked(False)
win.opacity.setValue(50); wait(lambda: False, 0.2)
check("opacity 50%", abs(L.mask[int(py * k), int(px * k)] - 0.5) < 0.06)
win.opacity.setValue(100)
win.feather.setValue(40); wait(lambda: False, 0.2)
check("feather softens the edge", 0.0 < float(L.mask.max()) <= 1.0)
win.feather.setValue(0)
win.layer_list.item(1).setCheckState(appmod.Qt.CheckState.Unchecked); wait(lambda: False, 0.2)
check("hidden layer isn't rendered", not win.view.stack)
win.layer_list.item(1).setCheckState(appmod.Qt.CheckState.Checked); wait(lambda: False, 0.2)
check("shown again", len(win.view.stack) == 1)
win._set_mask_tool(None)

# 5b. Copy layer: same selection and settings, above the original, then edited on its own
win.copy_layer(); wait(lambda: False, 0.3)
C = win.layers[1] if len(win.layers) == 2 else None
check("copy is added above and active", C is not None and win.active == 1 and C.name == "Layer 1 copy"
      and win.layer_list.item(2).text() == "Layer 1 copy" and win.layer_list.currentRow() == 2)
check("copy has the same selection and settings", C is not None and np.array_equal(C.mask, L.mask)
      and C.settings == L.settings and C.paint is not L.paint and len(win.view.stack) == 2)
win.sliders["exposure"][0].setValue(50); wait(lambda: False, 0.3)
check("editing the copy leaves the original", L.settings["development"]["exposure"] == -1.0
      and win._ui_settings()["development"]["exposure"] == 0.5)
win.delete_layer(); wait(lambda: False, 0.3)
win.layer_list.setCurrentRow(1); wait(lambda: False, 0.3)
check("copy deleted, original back", len(win.layers) == 1 and win.active == 0
      and win.sliders["exposure"][0].value() == -100)

# 6. AI per layer: noise reduction on the layer only
win.nr_group.setChecked(True)
t = wait()
dn_keys = [key for key in win.results if key[1] == "dn"]
check("layer's noise reduction runs (Base has it off)", len(dn_keys) == 1 and win.view.stack[0]["dn"] is not None
      and not win.settings()["enhancement"]["noise_reduction"]["enabled"], f"({t:.1f}s)")
win.nr_group.setChecked(False); wait()

# 7. Saving: the layer is in the file
f = out_dir / "layers_saved.png"; f.unlink(missing_ok=True)
with mock.patch.object(appmod.QFileDialog, "getSaveFileName", return_value=(str(f), "PNG 8-bit (smaller) (*.png)")):
    win.save_dialog()
wait(lambda: f.exists() and not win.saving, 180)
saved = cv2.imread(str(f))[..., ::-1].astype(np.float32) / 255
plain = np.clip(develop.develop(win.photo.linear, develop.DEFAULTS, win.photo.baseline_ev), 0, 1)
d_in = float(plain[int(py), int(px)].mean() - saved[int(py), int(px)].mean())
d_out = abs(float(saved[int(far[1]), int(far[0])].mean() - plain[int(far[1]), int(far[0])].mean()))
check("saved file has the layer's edit, only there", d_in > 0.05 and d_out < 0.02, f"(inside -{d_in:.3f}, outside {d_out:.3f})")
f.unlink(missing_ok=True)

# 8. Presets are the Base's; delete; a new photo drops layers
with mock.patch.object(appmod.QInputDialog, "getText", return_value=("Layer test", True)):
    win.save_preset()
check("preset saved from a layer holds the Base", presets.load("Layer test")["development"]["exposure"] == 0.0)
win.delete_layer(); wait(lambda: False, 0.3)
check("deleted: Base active, nothing stacked", win.active is None and not win.layers and not win.view.stack)
win.add_layer(); win._set_mask_tool(None)
with mock.patch.object(appmod.MainWindow, "_leave_photo", return_value=True):
    win.open_path(photo); wait(lambda: not win.loading)
check("new photo drops layers", not win.layers and win.layer_list.count() == 1)

import shutil; shutil.rmtree(presets.PRESET_DIR, ignore_errors=True)
win._leave_photo = lambda: True  # quitting asks to keep the edits
win.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
