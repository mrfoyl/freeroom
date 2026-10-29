# Freeroom

Simple RAW development + local AI noise reduction / sharpening on the GPU
(SCUNet and Restormer via spandrel + CUDA PyTorch), in its own `freeroom`
distrobox.

Three tabs, applied in this order:
1. **Raw**: exposure, shadows, midtones, highlights, contrast, temperature,
   tint, saturation. RAW files load scene-linear (rawpy, camera WB) with an
   automatic base exposure; Highlights left rolls RAW data past white back
   into range. Per-pixel only, so it's instant. A JPEG at defaults is unchanged.
   Below the dials, an **HDR tone mapping** section (off by default):
   single-photo local tone mapping. Compression pulls the large-scale
   brightness towards mid grey (shadows up, highlights down), Local detail
   scales the texture on top. Works from an edge-preserving low-res brightness
   map of the whole photo, so the loupe matches the result.
2. **B&W** (off by default): black & white with a colour mixer. Per colour
   (red, orange, yellow, green, aqua, blue, purple, magenta) a slider makes
   that colour lighter/darker, with **High** / **Low** underneath doing the
   same only in its bright / dark parts. Greys are never changed.
3. **Enhancement**: the AI steps below, shown on top of the development.
   Each section can be switched off.

Each AI model runs **at most once per photo** (when you first view it on the
Enhancement tab, or on save). The models work on a fixed encoding of the
linear photo, not on the developed one, so their results are kept and the
dials are applied on top afterwards: changing development, switching tabs,
toggling sections or going back to a model you already tried is instant.
Only a new photo (or a model not yet tried on this one) runs the AI again.
**Cancel** (next to the progress bar) stops a run after the current tile;
it then waits until you change a model or reopen the Enhancement tab.

The RGB range view (histogram) above the tabs shows the current result, with
clip markers and a note when a RAW has highlight data past white.

Presets (all settings of both tabs) are JSON files in
`~/.config/freeroom/presets/`. Saving works from either tab; it runs any
missing enhancement first.

## Requirements

### Compatibility

`./setup/install.sh` picks how to install by itself (see Usage). Status of
each platform:

| Platform | How it installs | PyTorch | Status |
|---|---|---|---|
| Debian 13 | native | CPU | ✅ Tested on a real desktop (KDE, Wayland) and in a clean container |
| Ubuntu 24.04 | native | CPU | ✅ Tested in a clean container |
| Fedora 43 | native | CPU | ✅ Tested in a clean container |
| Bazzite | Fedora distrobox | CUDA (NVIDIA) | ✅ The app was tested with an RTX 4080, driver 615; ⚠️ the current installer's distrobox route hasn't run yet |
| Other Linux with distrobox | Fedora distrobox | CUDA or CPU | ⚠️ Untested; same route as Bazzite |
| Linux with an NVIDIA GPU, native | native | CUDA | ⚠️ Untested |
| Linux with a supported AMD GPU | native or distrobox | ROCm 6.4 | ⚠️ Untested on a supported card (see AMD below) |
| macOS 14+, Apple Silicon | native + `Freeroom.app` | Metal (Apple GPU) | ⚠️ Untested on a real Mac (installer and app bundle checked in a simulated run) |
| macOS on Intel | — | — | ❌ Not supported (PyTorch 2.8 has no Intel Mac build); the installer stops with a message |
| Linux on ARM, Windows | — | — | ❌ Not supported |

"Clean container" = fresh minimal image, installed as a normal user with
sudo: the installer added the missing system packages, then the app started
both offscreen and on an X server (Xvfb, Qt's xcb plugin) and an AI model ran.
Wayland was only tested on the Debian desktop. Other apt- or dnf-based
distros (Linux Mint, Pop!_OS, RHEL/Alma/Rocky with EPEL…) should work the
same way; with any other package manager the installer can't add system
packages, but it says which libraries are missing.

### What's needed

- Linux: **x86_64** (the PyTorch and rawpy wheels used are x86_64; no ARM),
  a **Wayland or X11 desktop** (KDE, GNOME, …), and **sudo** for the system
  packages the installer may add (curl, Qt's X11/Wayland/GL libraries)
- macOS: **Apple Silicon** (M1 or newer; PyTorch 2.8 has no Intel Mac build)
  and **macOS 14 Sonoma or newer** (PySide6 6.12)
- Optional: **distrobox** with podman or docker. If it's installed, Freeroom
  goes in its own Fedora box and the host distro matters little; otherwise it
  installs natively (uv brings its own Python 3.12).
- For fast AI steps: an **NVIDIA GPU with the proprietary driver ≥ 570** (CUDA
  12.8 PyTorch build) or a **ROCm-supported AMD GPU** (Linux), and **≥ 4 GB
  VRAM** (each AI step uses ~1.3–3 GB)

The installer picks the PyTorch build: CUDA 12.8 when `nvidia-smi` finds a GPU,
ROCm 6.4 when it finds a supported AMD GPU, otherwise the CPU build (~1.7 GB
environment instead of ~7 GB). On the CPU the Raw and B&W tabs (HDR included)
are still instant, but the AI steps take minutes to hours per photo instead of
seconds.

**AMD (ROCm):** the installer reads each AMD GPU's target from sysfs (no ROCm
tools needed) and picks the ROCm build only for targets PyTorch's ROCm build
really runs on (its own kernels *and* rocBLAS's): gfx908, gfx90a, gfx942
(Instinct), gfx1030 (RX 6800/6900), gfx1100–1102 (RX 7000), gfx1200/1201
(RX 9000). Older Vega (gfx900/906) has PyTorch kernels but no rocBLAS, so it
can't work. Other AMD GPUs, including
integrated Radeon graphics, get the CPU build with a note; `--rocm` forces the
ROCm build, usually together with `HSA_OVERRIDE_GFX_VERSION` set to a close
supported target (hit and miss). You need to be in the `render` (and `video`)
group; the installer says so if you can't open `/dev/kfd`. With several AMD
GPUs Freeroom uses the first one the build supports, and with none it uses the
CPU instead of failing. Models run in float16 on AMD. The ROCm environment is
about **14 GB**. Linux only.

On a Mac the installer always installs natively with the `cpu` extra, which
there is the standard macOS PyTorch: the AI steps run on the Apple GPU through
Metal (in float32; ops Metal lacks fall back to the CPU). It also creates
`~/Applications/Freeroom.app`, which runs `freeroom.sh`; open photos with
Open… or by dropping them on the window (the app doesn't get Finder's
"Open with").

## Usage

- Setup (re-runnable): `./setup/install.sh`
  (`--box` / `--native` to choose where, `--cpu` / `--cu128` to force the build;
  the choice is saved in `.install`). A native Linux install first adds any missing
  system packages it needs (curl, Qt's X11/Wayland/GL libraries) with sudo, via apt
  or dnf, and warns afterwards if a library is still missing.
- Run: app menu → **Freeroom**, or `./freeroom.sh [photo]`
- Smoke test: `~/.local/bin/uv run --no-sync python tests/smoke.py`
  (inside the box when installed there: `distrobox enter freeroom -- bash -lc '…'`)
- Updating deps by hand: `uv sync --extra cpu` or `uv sync --extra cu128`. Plain
  `uv sync` or `uv run` without `--no-sync` would drop PyTorch, as it's an extra.

Loupe = 100% (1 photo px = 1 screen px). Hold mouse = original in loupe,
hold Space = original everywhere, scroll = loupe size.

Pipeline: denoise model → optional AI sharpen (deblur) model on the denoised
result, both in the AI encoding (`pipeline.py`). Each runs once per photo (RTX 4080, bf16, 512 px tiles: 12 MP ≈ 4 s
denoise + 9 s sharpen; 26 MP ≈ 9 s + 19 s). Of the deblur output only the
fine detail is kept (high-pass, σ=6) so it can't shift tones/colours.

Sliders only blend the AI results with the original (luminance / colour
noise separately, AI sharpen 0–150 %, plus a classic unsharp mask), so
they're instant.

Models (`setup/download-models.sh`):
- `models/denoise/`: SCUNet real psnr (default, faithful), gan (crisper, may invent texture)
- `models/sharpen/`: Restormer defocus deblurring (soft focus/lens) and motion deblurring (shake)

Any 1x RGB model spandrel (+ extra arches) can load can be dropped into either folder.

`tests/smoke.py [file.CR2]` = development, engine, presets, GUI flow (offscreen);
`tests/real_photo.py <photo>` = comparison crop on a real photo (read-only).

Saves go to `~/Pictures/freeroom/` by default; the original is never
overwritten. JPEG keeps EXIF + ICC; TIFF/PNG are 16-bit (no EXIF).
