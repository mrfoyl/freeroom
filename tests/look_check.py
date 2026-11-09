"""Looks (LUTs) and edits kept with the photo (sidecar).

    python tests/look_check.py <photo> [<another photo>]

The photos are copied to tests/out/look first (the sidecar is written next to
the copy), so the originals are only read. Everything is deleted afterwards.
"""
import os, shutil, sys, time
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["XDG_CONFIG_HOME"] = str(Path(__file__).parent / "out" / "config")  # never the user's real settings
from unittest import mock

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from freeroom import app as appmod, batch, develop, looks, presets, sidecar

out = Path(__file__).parent / "out" / "look"
shutil.rmtree(out, ignore_errors=True)
(out / "luts" / "film").mkdir(parents=True)
looks.LUT_DIR = out / "luts"
presets.PRESET_DIR = out / "presets"
fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name, extra)
    if not ok:
        fails.append(name)


def write_cube(path, n, fn, header=""):
    g = np.linspace(0, 1, n)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    rgb = np.stack(fn(r.ravel(), gg.ravel(), b.ravel()), 1)
    path.write_text(f"# test\nTITLE \"t\"\n{header}LUT_3D_SIZE {n}\n" + "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in rgb) + "\n")


# --- 1. reading and applying ------------------------------------------------------
write_cube(looks.LUT_DIR / "identity.cube", 17, lambda r, g, b: (r, g, b))
write_cube(looks.LUT_DIR / "film" / "cool.cube", 33, lambda r, g, b: (r * 0.8, g, b * 0.9 + 0.05))
write_cube(looks.LUT_DIR / "half_domain.cube", 9, lambda r, g, b: (r, g, b), "DOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n")
(looks.LUT_DIR / "invert1d.cube").write_text("LUT_1D_SIZE 2\n1 1 1\n0 0 0\n")
L = 4; n = L * L
hb, hg, hr = np.meshgrid(*[np.linspace(0, 1, n)] * 3, indexing="ij")
cv2.imwrite(str(looks.LUT_DIR / "hald.png"), (np.stack([hb, hg, hr], -1).reshape(L**3, L**3, 3) * 255 + .5).astype(np.uint8))
(looks.LUT_DIR / "broken.cube").write_text("LUT_3D_SIZE 4\n0 0 0\n")

x = np.random.default_rng(1).random((300, 400, 3), dtype=np.float32)
check("names list LUTs in subfolders", looks.names() == sorted(
    ["broken.cube", "film/cool.cube", "half_domain.cube", "hald.png", "identity.cube", "invert1d.cube"], key=str.lower),
    str(looks.names()))
check(".cube identity keeps the image", np.abs(looks.apply(x, looks.get("identity.cube")) - x).max() < 2e-4)
check("Hald CLUT identity keeps the image", np.abs(looks.apply(x, looks.get("hald.png")) - x).max() < 4e-3)
check("1D LUT inverts", np.abs(looks.apply(x, looks.get("invert1d.cube")) - (1 - x)).max() < 1e-5)
check("DOMAIN_MAX 2 halves the values", np.abs(looks.apply(x, looks.get("half_domain.cube")) - x / 2).max() < 2e-4)
cool = looks.apply(x, looks.get("film/cool.cube"))
exp = np.stack([x[..., 0] * .8, x[..., 1], x[..., 2] * .9 + .05], -1)
check("a look does what its table says", np.abs(cool - exp).max() < 1e-3, f"({np.abs(cool - exp).max():.4f})")
half = looks.apply(x, looks.get("film/cool.cube"), 0.5)
check("strength 50% is halfway", np.abs(half - (x + cool) / 2).max() < 1e-4)
check("broken file: a clear problem", "expected 64 table rows" in (looks.problem("broken.cube") or ""), looks.problem("broken.cube"))
check("missing file: a clear problem", "not found" in (looks.problem("nope.cube") or ""))
outside = out / "elsewhere.cube"; shutil.copy(looks.LUT_DIR / "film" / "cool.cube", outside)
check("settings name LUTs in the folder by name, others by path",
      looks.setting_name(looks.LUT_DIR / "film" / "cool.cube") == "film/cool.cube"
      and looks.setting_name(outside) == str(outside.resolve()))
t0 = time.perf_counter(); looks.apply(np.random.rand(1200, 1800, 3).astype(np.float32), looks.get("film/cool.cube"))
print(f"     (look on a 2.2 MP view: {1000 * (time.perf_counter() - t0):.0f} ms)")

# --- 2. development and batches -----------------------------------------------------
lin = develop.srgb_to_linear(x)
plain = develop.develop(lin, develop.DEFAULTS)
looked = develop.develop(lin, dict(develop.DEFAULTS, lut="film/cool.cube"))
check("develop applies the look last", np.abs(looked - looks.apply(plain, looks.get("film/cool.cube"))).max() < 1e-5)
check("strength 0 = no look", np.array_equal(develop.develop(lin, dict(develop.DEFAULTS, lut="film/cool.cube", lut_strength=0.0)), plain))
check("a missing LUT doesn't break developing", np.array_equal(develop.develop(lin, dict(develop.DEFAULTS, lut="gone.cube")), plain))
try:
    batch.job_from_settings({"development": {"lut": "gone.cube"}}, {})
    check("batch refuses a missing LUT", False)
except ValueError as e:
    check("batch refuses a missing LUT", "not found" in str(e), str(e))
check("batch takes a good LUT", batch.job_from_settings({"development": {"lut": "film/cool.cube"}}, {}).dev["lut"] == "film/cool.cube")

# --- 2b. exporting settings as a LUT ------------------------------------------------
neutral = develop.develop(lin, develop.DEFAULTS)
looks_p = dict(develop.DEFAULTS, exposure=0.3, shadows=0.2, highlights=-0.3, contrast=0.25, midtones=0.1,
               temperature=0.2, tint=-0.1, saturation=0.2, lut="film/cool.cube", lut_strength=0.6)
bw_p = dict(develop.DEFAULTS, bw=True, bw_red=0.5, bw_blue=-0.6, bw_green_hi=0.4, contrast=0.3)
for label, params, size, tol in (("colour settings", looks_p, 33, 0.004), ("B&W mix", bw_p, 65, 0.006)):
    f = looks.write_cube(out / f"export_{size}.cube", develop.look_table(params, size), "Test look")
    back = looks.read(f)
    via = looks.apply(neutral, back)
    err = np.abs(via - develop.develop(lin, params))
    check(f"exported LUT ({label}, {size}) = Freeroom's own result", back.table.shape == (size,) * 3 + (3,)
          and float(err.mean()) < tol and float(np.percentile(err, 99.9)) < 10 * tol,
          f"(mean {err.mean():.4f}, 99.9% {np.percentile(err, 99.9):.4f})")
check("default settings export as identity", np.abs(develop.look_table(develop.DEFAULTS, 17)
      - looks.read(looks.LUT_DIR / "identity.cube").table).max() < 1e-4)
check("HDR is left out of the LUT", np.array_equal(develop.look_table(dict(looks_p, hdr=True), 9), develop.look_table(looks_p, 9)))

# --- 3. the app: Look section, keep edits, reopen -----------------------------------------
src = [Path(p) for p in sys.argv[1:3]]
photos = []
for p in src:
    shutil.copy(p, out / p.name); photos.append(out / p.name)
a = photos[0]
b = photos[1] if len(photos) > 1 else None
qapp = QApplication([])
win = appmod.MainWindow(); win.resize(1500, 950); win.show()


def wait(cond, limit=60):
    t0 = time.time()
    while time.time() - t0 < limit and not cond():
        qapp.processEvents(); time.sleep(0.02)
    for _ in range(5):
        qapp.processEvents()


def opened(p):
    return win.photo is not None and win.photo.path == p and not win.loading


win.open_path(a); wait(lambda: opened(a))
check("LUT list: None + the folder's files", win.lut_box.count() == 7 and win.lut_box.itemData(0) == ""
      and win.lut_box.findData("film/cool.cube") > 0)
check("strength off without a LUT", not win.sliders["lut_strength"][0].isEnabled())
win.lut_box.setCurrentIndex(win.lut_box.findData("film/cool.cube")); wait(lambda: False, 0.3)
check("choosing a LUT sets it", win.dev_params["lut"] == "film/cool.cube" and win.view.dev_params["lut"] == "film/cool.cube"
      and win.sliders["lut_strength"][0].isEnabled())
win.lut_box.setCurrentIndex(win.lut_box.findData("broken.cube")); wait(lambda: False, 0.3)
check("a broken LUT says why", win.lut_note.isVisible() and "table rows" in win.lut_note.text(), win.lut_note.text())
with mock.patch.object(appmod.QFileDialog, "getOpenFileName", return_value=(str(outside), "")):
    win.pick_lut()
check("File…: a LUT from elsewhere, by path", win.dev_params["lut"] == str(outside.resolve())
      and win.lut_box.currentData() == str(outside.resolve()) and not win.lut_note.isVisible())
win.lut_box.setCurrentIndex(win.lut_box.findData("film/cool.cube"))
win.sliders["lut_strength"][0].setValue(60)
win.sliders["exposure"][0].setValue(30)
h, w = win.photo.linear.shape[:2]
win.view.set_crop((w // 10, h // 10, w - w // 10, h - h // 8))
win.add_layer(); win._set_mask_tool(None)
win.layers[0].stroke([(w * .3, h * .4), (w * .5, h * .5)], max(h, w) / 30, keep=True); win.layers[0].rebuild()
win.feather.setValue(12)
win.sliders["exposure"][0].setValue(-80)  # the layer's
win.set_active(None); wait(lambda: False, 0.3)
check("preset holds the look", (presets.save("Looky", win.settings()) and presets.load("Looky")["development"]["lut"] == "film/cool.cube"))
win.keep_edits()
car = sidecar.path_for(a)
check("Keep edits writes the sidecar next to the photo", car.is_file() and car.name == a.name + ".freeroom.json", win.status.text())
data = sidecar.read(a)
check("sidecar holds look, crop and layer", data["settings"]["development"]["lut"] == "film/cool.cube"
      and abs(data["settings"]["development"]["lut_strength"] - .6) < 1e-6 and data["crop"] and len(data["layers"]) == 1
      and data["layers"][0]["settings"]["development"]["exposure"] == -0.8)
kept_state = win.edit_state()
mask_before = win.layers[0].mask.copy()

with mock.patch.object(appmod.MainWindow, "_ask_keep", return_value="cancel") as q:
    other = b or a
    if b:
        win.open_path(b); wait(lambda: opened(b))
    else:  # reopen the same photo: leave first
        win._leave_photo(); win.photo = None; win.open_path(a); wait(lambda: opened(a) and win.layers)
check("kept edits: leaving doesn't ask", not q.called)
if b:
    check("the other photo opens plain", not win.layers and win.dev_params["lut"] == "" and win.view.crop is None)
    with mock.patch.object(appmod.MainWindow, "_ask_keep", return_value="cancel") as q:
        win.open_path(a); wait(lambda: opened(a))
    check("an untouched photo: no question", not q.called)
check("reopened: look back", win.dev_params["lut"] == "film/cool.cube" and win.lut_box.currentData() == "film/cool.cube"
      and win.sliders["lut_strength"][0].value() == 60 and win.sliders["exposure"][0].value() == 30)
check("reopened: crop and layer back", win.view.crop == tuple(data["crop"]) and len(win.layers) == 1
      and np.array_equal(win.layers[0].mask, mask_before) and win.layers[0].feather == 12 and len(win.view.stack) == 1)
check("reopened: exactly the kept edit", sidecar.same(a, win.edit_state()))

win.sliders["contrast"][0].setValue(20)
with mock.patch.object(appmod.MainWindow, "_ask_keep", return_value="cancel") as q:
    win.open_path(b or a); wait(lambda: False, 0.5)
check("changed after keeping: asks", q.called and opened(a))
with mock.patch.object(appmod.MainWindow, "_ask_keep", return_value="keep"):
    win.open_path(b or a); wait(lambda: opened(b) if b else True)
check("Keep with photo updates the sidecar", sidecar.read(a)["settings"]["development"]["contrast"] == 0.2)

# saved file has the look
if b:
    win.open_path(a); wait(lambda: opened(a))
f = out / "look_saved.png"
with mock.patch.object(appmod.QFileDialog, "getSaveFileName", return_value=(str(f), "PNG 8-bit (smaller) (*.png)")):
    win.save_dialog()
wait(lambda: f.exists() and not win.saving, 180)
saved = cv2.imread(str(f))[..., ::-1].astype(np.float32) / 255
x0, y0, x1, y1 = win.view.crop
dev = dict(win.settings()["development"])
ref = develop.develop(win.photo.linear[y0:y1, x0:x1], dev, win.photo.baseline_ev)
noref = develop.develop(win.photo.linear[y0:y1, x0:x1], dict(dev, lut=""), win.photo.baseline_ev)
far = (slice(0, (y1 - y0) // 8), slice(-(x1 - x0) // 8, None))  # corner away from the layer
d_look, d_plain = np.abs(saved[far] - ref[far]).mean(), np.abs(saved[far] - noref[far]).mean()
check("saved file has the look", d_look < 0.01 and d_plain > d_look * 2, f"(look {d_look:.4f}, without {d_plain:.4f})")

exp_f = out / "exported look"
win.hdr_group.setChecked(True)
win.lut_size_box.setCurrentIndex(win.lut_size_box.findData(65))
with mock.patch.object(appmod.QFileDialog, "getSaveFileName", return_value=(str(exp_f), "")):
    win.export_lut()
exp_c = exp_f.with_name(exp_f.name + ".cube")
ok = exp_c.is_file() and looks.read(exp_c).table.shape[0] == 65
check("Export as LUT writes the tabs' look", ok and np.abs(looks.read(exp_c).table
      - develop.look_table(win.dev_params, 65)).max() < 1e-5 and "without HDR" in win.status.text(), win.status.text())
win.hdr_group.setChecked(False)

with mock.patch.object(appmod.MainWindow, "_ask_keep", return_value="discard"):
    win.reset_all_settings(); win.view.set_crop(None)
    for _ in range(len(win.layers)):
        win.layer_list.setCurrentRow(1); win.delete_layer()
    win.keep_edits()
check("keeping no edits removes the sidecar", not car.exists())

win._leave_photo = lambda: True  # quitting asks to keep the edits
win.close()
shutil.rmtree(out, ignore_errors=True)
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
