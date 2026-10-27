# Freeroom

Simple RAW development + local AI noise reduction / sharpening on the GPU
(SCUNet and Restormer via spandrel + CUDA PyTorch), in its own `freeroom`
distrobox.

Two tabs, applied in this order:
1. **Raw development**: exposure, shadows, midtones, highlights, temperature,
   tint, saturation. RAW files load scene-linear (rawpy, camera WB) with an
   automatic base exposure; Highlights left rolls RAW data past white back
   into range. Per-pixel only, so it's instant. A JPEG at defaults is unchanged.
2. **Enhancement**: the AI steps below, run on the developed photo when you
   open the tab (or save). Each section can be switched off.

The RGB range view (histogram) above the tabs shows the current result, with
clip markers and a note when a RAW has highlight data past white.

Presets (all settings of both tabs) are JSON files in
`~/.config/freeroom/presets/`. Saving works from either tab; it runs any
missing enhancement first.

- Setup (on the Bazzite host, re-runnable): `./setup/create-distrobox.sh`
- Run: app menu → **Freeroom**, or `./freeroom.sh [photo]`
- Smoke test: `distrobox enter freeroom -- bash -lc 'cd ~/github_projects_workspace/freeroom && ~/.local/bin/uv run --no-sync python tests/smoke.py'`

Loupe = 100% (1 photo px = 1 screen px). Hold mouse = original in loupe,
hold Space = original everywhere, scroll = loupe size.

Pipeline: denoise model → optional AI sharpen (deblur) model on the denoised
result. Each runs once per photo (RTX 4080, bf16, 512 px tiles: 12 MP ≈ 4 s
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
