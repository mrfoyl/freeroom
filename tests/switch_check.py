"""Opening another photo: preset question, then everything reset; Reset development unchecks HDR.

    python tests/switch_check.py <photo A> <photo B>
"""
import os, sys, time, shutil
from pathlib import Path
from unittest import mock
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QMessageBox, QInputDialog
from freeroom import app as appmod, presets

a, b = Path(sys.argv[1]), Path(sys.argv[2])
presets.PRESET_DIR = Path(__file__).parent / "out" / "presets_switch"
shutil.rmtree(presets.PRESET_DIR, ignore_errors=True)
SB = QMessageBox.StandardButton
qapp = QApplication([])
win = appmod.MainWindow(); win.show()
def wait(cond, limit=60):
    t0 = time.time()
    while time.time() - t0 < limit and not cond():
        qapp.processEvents(); time.sleep(0.02)
    for _ in range(5): qapp.processEvents()
def opened(p): return win.photo is not None and win.photo.path == p and not win.loading
def tweak():
    win.sliders["exposure"][0].setValue(70); win.hdr_group.setChecked(True)
    win.sh_group.setChecked(True); win.bw_group.setChecked(True)
def is_default(): return win._same_settings(win.settings(), win.default_settings())

win.open_path(a); wait(lambda: opened(a))
win.open_path(b); wait(lambda: opened(b)); win.open_path(a); wait(lambda: opened(a))
print("no changes -> switches without asking:", opened(a))

tweak()
with mock.patch.object(QMessageBox, "question", return_value=SB.Cancel) as q:
    win.open_path(b); wait(lambda: False, 1)
print("Cancel -> stays on A with settings kept:", q.called, opened(a), not is_default())

with mock.patch.object(QMessageBox, "question", return_value=SB.Discard):
    win.open_path(b); wait(lambda: opened(b))
print("Discard -> B open, all defaults:", opened(b), is_default(), "| HDR", win.hdr_group.isChecked(),
      "B&W", win.bw_group.isChecked(), "sharpen", win.sh_group.isChecked(), "exposure", win.sliders["exposure"][0].value())

tweak()
with mock.patch.object(QMessageBox, "question", return_value=SB.Save), \
     mock.patch.object(QInputDialog, "getText", return_value=("Kept look", True)):
    win.open_path(a); wait(lambda: opened(a))
saved = presets.load("Kept look")
print("Save -> preset written with the tweaks:", saved["development"]["exposure"], saved["development"]["hdr"],
      "| then A open with defaults:", opened(a), is_default())

i = win.preset_box.findData("Kept look"); win.preset_box.setCurrentIndex(i); win.preset_chosen(i)  # = picking it
wait(lambda: True, 0.5)
with mock.patch.object(QMessageBox, "question", return_value=SB.Cancel) as q:
    win.open_path(b); wait(lambda: opened(b))
print("settings = selected preset -> no question, reset:", not q.called, opened(b), is_default())

with mock.patch.object(QMessageBox, "question", return_value=SB.Save), \
     mock.patch.object(QInputDialog, "getText", return_value=("", False)):
    tweak(); win.open_path(a); wait(lambda: False, 1)
print("Save but name dialog cancelled -> stays on B, settings kept:", opened(b), not is_default())

win.sliders["shadows"][0].setValue(40); win.hdr_group.setChecked(True)
win.reset_development()
print("Reset development -> HDR unchecked, sliders default:", not win.hdr_group.isChecked(),
      win.sliders["shadows"][0].value() == 0, win.sliders["exposure"][0].value() == 0)
win.close()
