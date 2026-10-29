# Freeroom

Simple RAW development + local AI noise reduction / sharpening on the GPU
(SCUNet and Restormer via spandrel + CUDA PyTorch), in its own `freeroom`
distrobox.

Four tabs, applied in this order:
1. **Raw**: exposure, shadows, midtones, highlights, contrast, temperature,
   tint, saturation. RAW files load scene-linear (rawpy, camera WB) with an
   automatic base exposure; Highlights left rolls RAW data past white back
   into range. Per-pixel only, so it's instant. A JPEG at defaults is unchanged.
2. **HDR** (off by default): single-photo local tone mapping. Compression pulls
   the large-scale brightness towards mid grey (shadows up, highlights down),
   Local detail scales the texture on top. Works from an edge-preserving
   low-res brightness map of the whole photo, so the loupe matches the result.
3. **B&W** (off by default): black & white with a colour mixer. Per colour
   (red, orange, yellow, green, aqua, blue, purple, magenta) a slider makes
   that colour lighter/darker, with **High** / **Low** underneath doing the
   same only in its bright / dark parts. Greys are never changed.
4. **Enhancement**: the AI steps below, shown on top of the development.
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

Tested on Bazzite (Fedora-based) with an RTX 4080 and NVIDIA driver 615 (in
the distrobox), and natively on Debian 13 without an NVIDIA GPU (CPU build).

- **x86_64** (the PyTorch and rawpy wheels used are x86_64; no ARM)
- **Wayland or X11 desktop** (KDE, GNOME, …)
- Optional: **distrobox** with podman or docker. If it's installed, Freeroom
  goes in its own Fedora box and the host distro matters little; otherwise it
  installs natively (uv brings its own Python 3.12).
- For fast AI steps: **NVIDIA GPU with the proprietary driver ≥ 570** (CUDA
  12.8 PyTorch build) and **≥ 4 GB VRAM** (each AI step uses ~1.3–3 GB)

The installer picks the PyTorch build: CUDA 12.8 when `nvidia-smi` finds a GPU,
otherwise the CPU build (~1.7 GB environment instead of ~7 GB). On the CPU the
Raw/HDR/B&W tabs are still instant, but the AI steps take minutes to hours
per photo instead of seconds. AMD GPUs would need the ROCm PyTorch build
instead (untested).

## Usage

- Setup (re-runnable): `./setup/install.sh`
  (`--box` / `--native` to choose where, `--cpu` / `--cu128` to force the build;
  the choice is saved in `.install`)
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
