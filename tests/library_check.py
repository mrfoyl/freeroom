"""Library view on a real folder (read-only): thumbnail speed, preview, open in editor.

    python tests/library_check.py <folder with photos>
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from freeroom import app as appmod, browser

folder = Path(sys.argv[1])
out = Path(__file__).parent / "out"; out.mkdir(exist_ok=True)
before = {e.name: (e.stat().st_mtime_ns, e.stat().st_size) for e in os.scandir(folder)}

qapp = QApplication([])
win = appmod.MainWindow(); win.resize(1500, 950); win.show()
lib = win.library
def wait(cond, limit=300):
    t0 = time.time()
    while time.time() - t0 < limit and not cond():
        qapp.processEvents(); time.sleep(0.02)
    for _ in range(5): qapp.processEvents()
    return time.time() - t0

def loaded():
    return all(not it.icon().pixmap(32).toImage().pixelColor(16, 16).name() == "#28282c" for it in lib.items.values())

print("starts in Library:", win.pages.currentIndex() == 0, "| Save hidden:", not win.save_btn.isVisible())
for run in ("first visit", "again (cached)"):
    lib.folder = None; t = time.time(); lib.show_folder(folder)
    n = len(lib.items); dt = wait(loaded)
    print(f"{run}: {n} photos, all thumbnails in {dt:.1f}s ({dt / max(n, 1) * 1000:.0f} ms each)")
lib.grid.setCurrentIndex(lib.model.index(min(3, len(lib.items) - 1), 0))
wait(lambda: lib.preview.img.width() > browser.THUMB_SIZE, 30)
print("preview:", lib.preview.img.width(), "x", lib.preview.img.height(), "|", lib.info.text().replace("<br>", " | ").replace("<b>", "").replace("</b>", ""))
win.grab().save(str(out / "library.png"))
# orientation: thumbnail aspect vs real pixels for every photo
rot = []
for path, it in lib.items.items():
    im = it.icon().pixmap(browser.THUMB_SIZE).toImage()
    rot.append((Path(path).name, "portrait" if im.height() > im.width() else "landscape"))
print("portrait thumbs:", [n for n, o in rot if o == "portrait"][:8])
lib._open_index(lib.grid.currentIndex())
wait(lambda: win.photo is not None and not win.loading, 60)
print("opened in editor:", win.pages.currentIndex() == 1, win.photo.path.name if win.photo else None)
win.set_mode(edit=False)
print("back in Library, same photo selected:", Path(lib.grid.currentIndex().data(browser.PATH_ROLE)).name)
after = {e.name: (e.stat().st_mtime_ns, e.stat().st_size) for e in os.scandir(folder)}
print("source folder untouched:", before == after)
win.close()

# RAW only checkbox (remembered)
qapp2 = qapp
win = appmod.MainWindow(); win.show(); lib = win.library
lib.show_folder(folder); total = len(lib.items)
lib.raw_only.setChecked(True); wait(lambda: True, 1)
raws = [Path(p).suffix.lower() for p in lib.items]
print(f"RAW only: {len(raws)} of {total} shown, all RAW: {all(e in __import__('freeroom').imageio.RAW_EXTS for e in raws)}, count label: {lib.count_label.text()}")
win.close(); win = appmod.MainWindow(); print("remembered after restart:", win.library.raw_only.isChecked(), len(win.library.items))
win.library.raw_only.setChecked(False); print("off again:", len(win.library.items) == total); win.close()

# shooting info: Raw tab while editing + Library preview
win = appmod.MainWindow(); win.resize(1500, 950); win.show(); lib = win.library
lib.show_folder(folder)
raw = next(p for p in lib.items if Path(p).suffix.lower() in __import__('freeroom').imageio.RAW_EXTS)
lib.select(Path(raw)); wait(lambda: "ISO" in lib.info.text(), 30)
print("library preview info:", lib.info.text().replace("<br>", " | ").replace("<b>", "").replace("</b>", ""))
lib._open_index(lib.grid.currentIndex()); wait(lambda: win.photo is not None and not win.loading, 60)
print("Raw tab:", win.shot_head.text(), "|", __import__('re').sub("<[^>]+>", " ", win.shot_rows.text()).split())
print("view rendered right after load:", win.view._view_img is not None)
t = wait(lambda: win.view._view_img is not None, 20); print(f"view rendered after {t:.2f}s more:", win.view._view_img is not None)
win.grab().save(str(out / "raw_tab_info.png")); win.close()
