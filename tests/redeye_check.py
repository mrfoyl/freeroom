"""Red eye: finding by itself and by hand, driven with real mouse clicks (offscreen)."""
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
    cv2.circle(img, (ex, ey), PUPIL, (205, 34, 42), -1)                     # red pupil
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
    t = np.full((1200, 1200, 3), (205, 165, 145), np.uint8); cv2.circle(t, (600, 600), pr, (205, 34, 42), -1)
    t = cv2.GaussianBlur(t, (0, 0), 1.0); ll = develop.srgb_to_linear(t.astype(np.float32) / 255)
    f = redeye.find(ll, 600 + min(pr // 2, 40), 600, 57 if pr < 150 else 100); print(f"  pupil radius {pr}: found radius {f.radius:.1f}" if f else f"  pupil radius {pr}: NOT FOUND")

# --- a dark maroon pupil with a bright red fragment: the mask has to cover the whole pupil
t = np.full((800, 800, 3), (205, 165, 145), np.uint8); cv2.circle(t, (400, 400), 110, (70, 110, 150), -1)
cv2.circle(t, (400, 400), 50, (72, 14, 20), -1); cv2.circle(t, (415, 425), 12, (200, 30, 40), -1)
tl = develop.srgb_to_linear(cv2.GaussianBlur(t, (0, 0), 1.0).astype(np.float32) / 255); f = redeye.find(tl, 415, 425, 40)
print(f"maroon pupil radius 50: mask radius {f.radius:.1f}" if f else "maroon pupil: NOT FOUND")
# a red fragment on reddish skin that has green below blue: must not run out over the face
t = np.full((800, 800, 3), (200, 130, 140), np.uint8); cv2.circle(t, (400, 400), 12, (200, 30, 40), -1)
tl = develop.srgb_to_linear(cv2.GaussianBlur(t, (0, 0), 1.0).astype(np.float32) / 255); f = redeye.find(tl, 400, 400, 40)
print(f"fragment radius 12 on pinkish skin: mask radius {f.radius:.1f}" if f else "fragment: NOT FOUND")

# --- the other eye from a found one, to both sides
for me, other in ((0, 1), (1, 0)):
    s0 = redeye.find(lin, *EYES[me], 90); s1 = redeye.partner(lin, s0)
    print(f"partner of eye {me}: off by {np.hypot(s1.cx - EYES[other][0], s1.cy - EYES[other][1]):.1f}px" if s1 else f"partner of eye {me}: NOT FOUND")
t = img.copy(); cv2.circle(t, EYES[1], 115, (70, 110, 150), -1)  # other eye not red
print("no red partner:", redeye.partner(develop.srgb_to_linear(t.astype(np.float32) / 255), redeye.find(lin, *EYES[0], 90)))
for name, r2, dx in (("half the size", 25, 1200), ("much too close", 55, 300), ("much too far", 55, 3800)):
    t = np.full((3000, 6000, 3), (205, 165, 145), np.uint8); cv2.circle(t, (1000, 1500), 55, (205, 34, 42), -1); cv2.circle(t, (1000 + dx, 1500), r2, (205, 34, 42), -1)
    tl = develop.srgb_to_linear(cv2.GaussianBlur(t, (0, 0), 1.0).astype(np.float32) / 255)
    print(f"  partner {name}:", redeye.partner(tl, redeye.find(tl, 1000, 1500, 90)))

# --- searching a whole photo (a real face, public domain NASA portrait, with red pupils painted on)
sample = Path(__file__).parent / "out" / "samples" / "astronaut.png"
if not sample.exists():
    sample.parent.mkdir(exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve("https://raw.githubusercontent.com/scikit-image/scikit-image/v0.22.0/skimage/data/astronaut.png", sample)
face = cv2.cvtColor(cv2.imread(str(sample)), cv2.COLOR_BGR2RGB)
face = cv2.resize(face, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)  # 3072 px
detector = redeye._Faces(develop.srgb_to_linear(face.astype(np.float32) / 255), 0.0)
detector.detector.setInputSize((768, 768))
small = cv2.resize(face, (768, 768), interpolation=cv2.INTER_AREA)
_, found = detector.detector.detect(small[..., ::-1])
detector.detector.setInputSize((redeye.FACE_SIZE, redeye.FACE_SIZE))
F_EYES = (found[0][4:8].reshape(2, 2) * 4).round().astype(int)
dist_eyes = float(np.hypot(*(F_EYES[0] - F_EYES[1])))
PR = max(int(dist_eyes / 16), 3)
print(f"real face: eyes {F_EYES.tolist()}, {dist_eyes:.0f} px apart, painted pupil radius {PR}")
def paint(base, eyes, r=PR):
    t = base.copy()
    for ex, ey in eyes:
        cv2.circle(t, (int(ex), int(ey)), r, (215, 28, 36), -1)
    return develop.srgb_to_linear(cv2.GaussianBlur(t, (0, 0), 1.0).astype(np.float32) / 255)
def got(linear, eyes, tol=None):
    spots = redeye.detect(linear); tol = tol or PR
    hit = sum(any(np.hypot(s.cx - e[0], s.cy - e[1]) < tol for s in spots) for e in eyes)
    return f"{hit} of {len(eyes)} eyes, {len(spots)} spots"
t0 = time.time(); print("both eyes:", got(paint(face, F_EYES), F_EYES), f"({time.time() - t0:.2f} s)")
print("no red eyes, plain face:", got(develop.srgb_to_linear(face.astype(np.float32) / 255), []))
print("one eye only:", got(paint(face, F_EYES[:1]), F_EYES[:1]))
for k in (0.5, 0.25):  # smaller in the frame
    sm = cv2.resize(face, None, fx=k, fy=k, interpolation=cv2.INTER_AREA); e = (F_EYES * k).round().astype(int)
    print(f"smaller face ({k}):", got(paint(sm, e, max(int(PR * k), 3)), e, max(PR * k, 4)))
for deg in (20, -35, 90):  # head tilted, or a portrait photo lying sideways
    c = (face.shape[1] / 2, face.shape[0] / 2); m = cv2.getRotationMatrix2D(c, deg, 1.0)
    tf = cv2.warpAffine(face, m, face.shape[1::-1]); e = (np.c_[F_EYES, np.ones(2)] @ m.T).round().astype(int)
    print(f"tilted {deg}:", got(paint(tf, e), e))
wide = np.zeros((3000, 8000, 3), np.uint8); wide[:] = (60, 60, 65)  # three people
for i, x in enumerate((100, 2600, 5000)):
    wide[0:2800, x:x + 2800] = face[0:2800, :2800]
e3 = [(int(ex + x), int(ey)) for x in (100, 2600, 5000) for ex, ey in F_EYES]
print("three faces:", got(paint(wide, e3), e3))

# things that aren't eyes: pairs of red round lights, a red pattern, a red shape on the face
plain = np.full((2000, 3000, 3), (70, 75, 80), np.uint8)
lights = plain.copy(); cv2.circle(lights, (1000, 1000), 30, (220, 26, 34), -1); cv2.circle(lights, (1500, 1000), 30, (220, 26, 34), -1)
ll = develop.srgb_to_linear(cv2.GaussianBlur(lights, (0, 0), 1.0).astype(np.float32) / 255)
print("two red lights, no face:", len(redeye.detect(ll)), "spots")
sl = redeye.find(ll, 1000, 1000, 50); print("  ...a click there still fixes the pair:", redeye.partner(ll, sl) is not None)
dots = plain.copy()
for y in range(100, 1950, 90):
    for x in range(100, 2950, 90):
        cv2.circle(dots, (x, y), 14, (220, 26, 34), -1)
t0 = time.time(); print("red pattern:", len(redeye.detect(develop.srgb_to_linear(cv2.GaussianBlur(dots, (0, 0), 1.0).astype(np.float32) / 255))), f"spots, {time.time() - t0:.2f} s")
mouth = (F_EYES.mean(axis=0) + (0, dist_eyes)).round().astype(int)
print("red dot on the mouth:", got(paint(face, [mouth]), []))

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
win.tabs.setCurrentWidget(win.enh_tab); wait(lambda: not win.busy and win.results, 120); keep = dict(win.results)
def click(x, y, button=Qt.MouseButton.LeftButton):
    pt = v._to_widget(x, y); QTest.mouseMove(v, QPoint(round(pt.x()), round(pt.y()))); QTest.mouseClick(v, button, pos=QPoint(round(pt.x()), round(pt.y()))); wait(limit=0.2)
def shown_at(x, y):
    QTest.mouseMove(v, QPoint(5, 5)); wait(limit=0.4); pt = v._to_widget(x, y); c = v._view_img.pixelColor(round((pt.x() - v._rect.x()) * v.devicePixelRatioF()), round((pt.y() - v._rect.y()) * v.devicePixelRatioF()))
    return c.red(), c.green(), c.blue()
print("section is on the Enhancement tab:", win.enh_tab.isAncestorOf(win.re_group), "| off by default:", not win.re_group.isChecked(), "| hand button greyed:", not win.redeye_btn.isEnabled() or not win.re_group.isChecked())
print("top bar is free of it:", all(not w.isAncestorOf(win.redeye_btn) for w in win.edit_only))
print("pupil off:", shown_at(ex - 25, ey + 20))
win.re_group.setChecked(True); wait(lambda: not win.redeye_searching, 30); print("section on (cartoon, no real face):", len(v.spots), "|", win.redeye_status.text())
win.redeye_btn.click(); print("hand mode:", v.redeye_mode)
click(*EYES[0]); print("one click fixes both:", len(v.spots), "|", v.redeye_note[:30]); wait(limit=0.3); win.grab().save(str(out / "redeye_gui.png")); print("window width stays:", win.width())
print("pupils as shown:", shown_at(ex - 25, ey + 20), shown_at(EYES[1][0] - 25, EYES[1][1] + 20))
click(*EYES[1]); print("click on a fixed eye adds none:", len(v.spots), "|", v.redeye_note[:15])
click(ex, ey + 500); print("click on skin:", len(v.spots), "|", v.redeye_note[:20])
win.redeye_undo.click(); wait(limit=0.2); print("undo:", len(v.spots))
click(*EYES[1]); print("again:", len(v.spots))
click(*EYES[1], Qt.MouseButton.RightButton); print("right-click removes:", len(v.spots)); click(*EYES[0]); print("again both:", len(v.spots))
QTest.keyClick(v, Qt.Key.Key_Return); print("Enter leaves the mode:", not v.redeye_mode)
print("AI not re-run by a fix:", not win.busy, all(win.results[k] is keep[k] for k in keep), "results", len(win.results))
win.re_group.setChecked(False); wait(limit=0.2); print("off: no fixes:", len(v.spots)); win.re_group.setChecked(True); wait(limit=0.2); print("on again, fixes back:", len(v.spots))
def save(name):
    f = out / name; f.unlink(missing_ok=True)
    with mock.patch.object(appmod.QFileDialog, "getSaveFileName", return_value=(str(f), "")): win.save_dialog()
    wait(lambda: f.exists() and not win.saving, 60); return cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
a = save("redeye_saved.png"); print("saved full, pupils RGB:", [a[y + 20, x - 25].tolist() for x, y in EYES], "size", a.shape[:2])
v.set_crop((1000, 1000, 3000, 2000)); wait(limit=0.3)
a = save("redeye_saved_crop.png"); print("saved crop, pupils RGB:", [a[y + 20 - 1000, x - 25 - 1000].tolist() for x, y in EYES], "size", a.shape[:2])
v.set_crop(None)
win.tabs.setCurrentIndex(0); wait(limit=0.5); print("Raw tab shows no fix, like the AI steps (saved file has it):", shown_at(ex - 25, ey + 20)); print("hand mode off there:", not v.redeye_mode)
s_on = win.settings(); print("in settings:", s_on["enhancement"]["red_eye"], "| not default:", not win._same_settings(s_on, win.default_settings()))
win.tabs.setCurrentWidget(win.enh_tab)

# a real face in the app: found by itself, and saved even when saved right away
fsrc = out / "redeye_face.png"; cv2.imwrite(str(fsrc), cv2.cvtColor((np.clip(paint(face, F_EYES), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
win._leave_photo = lambda: True
win.open_path(fsrc); wait(lambda: win.photo.path == fsrc and not win.loading and win.redeye_found is not None, 30)
print("next photo, section still on (reset is _leave_photo's):", len(v.spots), "|", win.redeye_status.text())
win.redeye_undo.click(); print("undo one:", len(v.spots), "|", win.redeye_status.text()); win.redeye_again.click(); print("find again:", len(v.spots))
from freeroom import batch as batchmod
print("batch job carries it:", batchmod.job_from_settings(win.settings(), {k: 0.5 for k in appmod.ENH_KEYS}).redeye, "| older preset without it:", batchmod.job_from_settings({}, {k: 0.5 for k in appmod.ENH_KEYS}).redeye)
win.re_group.setChecked(False); win.reset_all_settings()
print("reset switches it off:", not win.re_group.isChecked(), "| no fixes:", len(v.spots))
win.close()
