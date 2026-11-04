"""Red eye tool on a synthetic portrait, driven with real mouse clicks (offscreen)."""
import os, time
from pathlib import Path
from unittest import mock
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["XDG_CONFIG_HOME"] = str(Path(__file__).parent / "out" / "config")  # never the user's real settings
import numpy as np, cv2
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog
from freeroom import app as appmod, develop, imageio, redeye

out = Path(__file__).parent / "out"; out.mkdir(exist_ok=True)
H, W = 3000, 4000
rng = np.random.default_rng(1)
img = np.full((H, W, 3), (205, 165, 145), np.uint8)                    # skin (RGB)
cv2.ellipse(img, (2000, 2300), (420, 110), 0, 0, 360, (190, 60, 70), -1)  # red lips
EYES = [(1400, 1300), (2600, 1300)]; PUPIL = 55
for ex, ey in EYES:
    cv2.ellipse(img, (ex, ey), (260, 130), 0, 0, 360, (235, 232, 228), -1)  # sclera
    cv2.circle(img, (ex, ey), 115, (70, 110, 150), -1)                      # iris
    cv2.circle(img, (ex, ey), PUPIL, (205, 40, 35), -1)                     # red pupil
    cv2.circle(img, (ex + 15, ey - 18), 12, (255, 255, 255), -1)            # catchlight
img = cv2.GaussianBlur(img, (0, 0), 1.5)
img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
src = out / "redeye_portrait.png"
cv2.imwrite(str(src), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# --- detection and correction on the photo itself
photo = imageio.load(src); lin = photo.linear
ex, ey = EYES[0]
t0 = time.time(); s = redeye.find(lin, ex + 30, ey - 20, 90); dt = time.time() - t0
print(f"found: centre off by {np.hypot(s.cx - ex, s.cy - ey):.1f}px, radius {s.radius:.1f} (real {PUPIL}), {dt * 1000:.0f} ms")
fixed = redeye.apply(lin, (s,))
print("input not modified:", fixed is not lin and not np.shares_memory(fixed, lin))
before, after = develop.develop(lin, develop.DEFAULTS), develop.develop(fixed, develop.DEFAULTS)
def px(a, x, y, r=8): return (a[y - r:y + r, x - r:x + r].mean(axis=(0, 1)) * 255).round().astype(int)
print("pupil before", px(before, ex - 25, ey + 20), "after", px(after, ex - 25, ey + 20))
yy, xx = np.mgrid[ey - 150:ey + 150, ex - 150:ex + 150]; dist = np.hypot(xx - ex, yy - ey); crop_b, crop_a = before[ey - 150:ey + 150, ex - 150:ex + 150], after[ey - 150:ey + 150, ex - 150:ex + 150]
for a0, a1 in ((0, 45), (45, 52), (52, 56), (56, 60), (60, 66), (66, 80), (80, 110)):
    ring = (dist >= a0) & (dist < a1); print(f"  radius {a0:>3}-{a1:<3} before {(crop_b[ring].mean(0) * 255).round().astype(int)} after {(crop_a[ring].mean(0) * 255).round().astype(int)}")
print("catchlight before", px(before, ex + 15, ey - 18, 3), "after", px(after, ex + 15, ey - 18, 3))
for name, (x, y) in {"iris": (ex + 85, ey), "sclera": (ex + 200, ey), "skin": (ex, ey + 400), "lips": (2000, 2300)}.items():
    print(f"  {name}: changed by {np.abs(after[y - 8:y + 8, x - 8:x + 8] - before[y - 8:y + 8, x - 8:x + 8]).max() * 255:.2f}/255")
yy, xx = np.mgrid[0:H, 0:W]; outside = np.hypot(xx - ex, yy - ey) > 115
print("changed pixels outside the iris:", int((np.abs(after - before).max(axis=2)[outside] > 1 / 255).sum()))
print("no false finds: skin", redeye.find(lin, ex, ey + 500, 90), "| white of eye", redeye.find(lin, ex + 200, ey, 40), "| big search on lips", redeye.find(lin, 2000, 2300, 90))
# same result at preview scale
small = cv2.resize(lin, (1000, 750), interpolation=cv2.INTER_AREA)
sf = redeye.apply(small, (s,), (0, 0, H, W)); mx, my = (ex - 25) // 4, (ey + 20) // 4
print("preview scale pupil:", (develop.develop(sf, develop.DEFAULTS)[my, mx] * 255).round().astype(int))
region = (1000, 1000, 1700, 1800)
cf = redeye.apply(lin[1000:1700, 1000:1800], (s,), region)
print("crop region identical to full:", np.array_equal(cf, fixed[1000:1700, 1000:1800]))

# imprecise clicks and other pupil sizes (default search radius of this photo = 57)
hits = sum(redeye.find(lin, ex + dx, ey + dy, 57) is not None and abs(redeye.find(lin, ex + dx, ey + dy, 57).radius - PUPIL) < 2
           for dx in (-50, -25, -3, 0, 3, 25, 50) for dy in (-50, -25, -3, 0, 3, 25, 50) if np.hypot(dx, dy) <= 50)
total = sum(1 for dx in (-50, -25, -3, 0, 3, 25, 50) for dy in (-50, -25, -3, 0, 3, 25, 50) if np.hypot(dx, dy) <= 50)
print(f"off-centre clicks finding the whole pupil: {hits} of {total}")
print("click outside the search circle:", redeye.find(lin, ex + 200, ey + 200, 57))
for pr in (6, 12, 25, 120, 170):
    t = np.full((1200, 1200, 3), (205, 165, 145), np.uint8); cv2.circle(t, (600, 600), pr, (205, 40, 35), -1)
    t = cv2.GaussianBlur(t, (0, 0), 1.0); ll = develop.srgb_to_linear(t.astype(np.float32) / 255)
    f = redeye.find(ll, 600 + min(pr // 2, 40), 600, 57 if pr < 150 else 100); print(f"  pupil radius {pr}: found radius {f.radius:.1f}" if f else f"  pupil radius {pr}: NOT FOUND")

# --- the app, with real clicks
qapp = QApplication([])
win = appmod.MainWindow(); win.resize(1500, 950); win.show()
def wait(cond=lambda: False, limit=120):
    t0 = time.time()
    while time.time() - t0 < limit and not cond():
        qapp.processEvents(); time.sleep(0.02)
    for _ in range(5): qapp.processEvents()
win.open_path(src); wait(lambda: win.photo is not None and not win.loading); wait(lambda: win.view._view_img is not None, 5)
v = win.view
def click(x, y, button=Qt.MouseButton.LeftButton):
    pt = v._to_widget(x, y); QTest.mouseMove(v, QPoint(round(pt.x()), round(pt.y()))); QTest.mouseClick(v, button, pos=QPoint(round(pt.x()), round(pt.y()))); wait(limit=0.2)
print("buttons before Red eye: Undo hidden", not win.redeye_undo.isVisible(), "| Clear hidden", not win.redeye_clear.isVisible())
win.redeye_btn.click(); print("Red eye mode on:", v.redeye_mode, "| Undo shown", win.redeye_undo.isVisible())
click(*EYES[0]); click(*EYES[1]); print("two clicks:", len(v.spots), "fixes,", win.redeye_btn.text())
wait(limit=0.5); win.grab().save(str(out / "redeye_gui.png")); print("window width stays:", win.width())
QTest.mouseMove(v, QPoint(5, 5)); wait(limit=0.3); pt = v._to_widget(ex - 25, ey + 20); shown = v._view_img.pixelColor(round((pt.x() - v._rect.x()) * v.devicePixelRatioF()), round((pt.y() - v._rect.y()) * v.devicePixelRatioF()))
print("pupil as shown in the app view:", shown.red(), shown.green(), shown.blue())
click(ex, ey + 500); print("click on skin:", len(v.spots), "|", v.redeye_note[:40])
win.redeye_undo.click(); wait(limit=0.2); print("undo:", len(v.spots)); click(*EYES[1]); print("again:", len(v.spots))
click(*EYES[1], Qt.MouseButton.RightButton); print("right-click removes:", len(v.spots)); click(*EYES[1])
QTest.keyClick(v, Qt.Key.Key_Return); print("Enter leaves the mode:", not v.redeye_mode, "| Clear still shown:", win.redeye_clear.isVisible())
win.tabs.setCurrentWidget(win.enh_tab); wait(lambda: not win.busy and win.results, 120); keep = dict(win.results)
v.set_spots(v.spots[:1]); wait(limit=0.3); win.redeye_btn.click(); click(*EYES[1]); QTest.keyClick(v, Qt.Key.Key_Escape)
print("AI not re-run by a fix:", not win.busy, all(win.results[k] is keep[k] for k in keep), "results", len(win.results))
def save(name):
    f = out / name; f.unlink(missing_ok=True)
    with mock.patch.object(appmod.QFileDialog, "getSaveFileName", return_value=(str(f), "")): win.save_dialog()
    wait(lambda: f.exists() and not win.saving, 60); return cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
a = save("redeye_saved.png"); print("saved full, pupils RGB:", [a[y + 20, x - 25].tolist() for x, y in EYES], "size", a.shape[:2])
v.set_crop((1000, 1000, 3000, 2000)); wait(limit=0.3)
a = save("redeye_saved_crop.png"); print("saved crop, pupils RGB:", [a[y + 20 - 1000, x - 25 - 1000].tolist() for x, y in EYES], "size", a.shape[:2])
win.tabs.setCurrentIndex(0); wait(lambda: win.view._view_img is not None, 2); v.set_crop(None)
win._leave_photo = lambda: True
win.open_path(out / "noisy.jpg"); wait(lambda: win.photo.path.name == "noisy.jpg" and not win.loading)
print("next photo starts with no fixes:", len(v.spots) == 0, "| Clear hidden:", not win.redeye_clear.isVisible(), "|", win.redeye_btn.text())
win.close()
