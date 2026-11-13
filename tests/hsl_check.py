"""Colour mixer: hue / saturation / luminance per colour, and its folded section.

    python tests/hsl_check.py

Synthetic colour patches; the GUI part runs offscreen with its own settings.
"""
import colorsys, os, sys
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["XDG_CONFIG_HOME"] = str(Path(__file__).parent / "out" / "config")  # never the user's real settings
import numpy as np

from freeroom import develop

fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name, extra)
    if not ok:
        fails.append(name)


# display colours -> linear, so development at defaults gives them back
patches = {"red": (0.8, 0.15, 0.1), "blue": (0.2, 0.2, 0.85), "green": (0.2, 0.7, 0.25), "grey": (0.5, 0.5, 0.5)}
disp = np.array([[c for c in patches.values()]], np.float32)
lin = develop.srgb_to_linear(disp)


def dev(**sliders):
    p = dict(develop.DEFAULTS)
    p.update(sliders)
    return dict(zip(patches, develop.develop(lin, p)[0]))


def hsv(c):
    return colorsys.rgb_to_hsv(*map(float, c))


base = dev()
check("defaults change nothing", np.allclose(np.array(list(base.values())), disp[0], atol=2e-3))

out = dev(hsl_red_hue=1.0)
h0, h1 = hsv(base["red"])[0] * 360, hsv(out["red"])[0] * 360
check("red hue +100 turns red towards orange", 20 < h1 - h0 < 32, f"({h0:.0f}° -> {h1:.0f}°)")
check("... value and saturation kept", abs(hsv(out["red"])[2] - hsv(base["red"])[2]) < 0.01
      and abs(hsv(out["red"])[1] - hsv(base["red"])[1]) < 0.02)
check("... blue, green and grey untouched", all(np.allclose(out[k], base[k], atol=1e-4) for k in ("blue", "green", "grey")))

out = dev(hsl_blue_sat=-1.0)
check("blue saturation -100 makes blue grey", np.ptp(out["blue"]) < 0.01, str(np.round(out["blue"], 3)))
check("... red untouched", np.allclose(out["red"], base["red"], atol=1e-4))
out = dev(hsl_green_sat=0.5)
check("green saturation +50 makes green stronger", hsv(out["green"])[1] > hsv(base["green"])[1] + 0.05)

out = dev(hsl_blue_lum=-1.0)
check("blue luminance -100 darkens blue", out["blue"].mean() < base["blue"].mean() * 0.8
      and abs(hsv(out["blue"])[0] - hsv(base["blue"])[0]) < 0.01)
check("... grey untouched", np.allclose(out["grey"], base["grey"], atol=1e-4))

out = dev(hsl_red_hue=1.0, bw=True)
check("black & white ignores the colour mixer", np.allclose(out["red"], dev(bw=True)["red"], atol=1e-5))

lut = develop.look_table({**develop.DEFAULTS, "hsl_red_hue": 1.0}, size=9)
check("colour mixer goes into exported LUTs", not np.allclose(lut, develop.look_table(dict(develop.DEFAULTS), size=9)))

# --- the folded section ------------------------------------------------
from PySide6.QtWidgets import QApplication
from freeroom import app as appmod
qapp = QApplication([])
win = appmod.MainWindow(); win.resize(1500, 950); win.show(); win.set_mode(edit=True)
for _ in range(5):
    qapp.processEvents()
check("Colour mixer folded at start", not win.hsl_toggle.isChecked() and win.hsl_body.isHidden()
      and win.hsl_toggle.text() == "Colour mixer")
win.hsl_toggle.click(); qapp.processEvents()
check("header click opens it", not win.hsl_body.isHidden() and win.hsl_pages.currentIndex() == 0)
win.sliders["hsl_red_hue"][0].setValue(50)
check("hue slider sets the development", win.dev_params["hsl_red_hue"] == 0.5
      and win.sliders["hsl_red_hue"][1].text() == "+15°")
check("header says it's in use", "in use" in win.hsl_toggle.text())
s = win.settings()["development"]
check("saved in settings / presets", s["hsl_red_hue"] == 0.5)
win.reset_development(); qapp.processEvents()
check("Reset development resets the mixer", win.dev_params["hsl_red_hue"] == 0.0 and win.hsl_toggle.text() == "Colour mixer")
win.apply_settings({"development": {"hsl_blue_lum": -0.3}})
check("presets restore it", win.sliders["hsl_blue_lum"][0].value() == -30)
win.hsl_toggle.click(); qapp.processEvents()
check("folds again", win.hsl_body.isHidden())
win.grab().save(str(Path(__file__).parent / "out" / "hsl_folded.png"))
win.hsl_toggle.click(); win.hsl_pages.setCurrentIndex(2); qapp.processEvents()
win.grab().save(str(Path(__file__).parent / "out" / "hsl_open.png"))
win.close()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
