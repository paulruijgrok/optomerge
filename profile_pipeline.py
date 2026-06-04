#!/usr/bin/env python3
"""
profile_pipeline.py
===================
Time every stage of the optomerge pipeline on one real movie and print a
ranked breakdown of where the time is spent.

Usage
-----
    python profile_pipeline.py "test_data/MyLOVChar4/ch3/movie 01.tif"

    # Use only the first 100 frames (faster profiling run)
    python profile_pipeline.py "test_data/MyLOVChar4/ch3/movie 01.tif" --frames 100

The script does NOT write any output files — it only measures time.
"""

from __future__ import annotations
import argparse, sys, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np

_here = Path(__file__).resolve().parent
_pkg  = _here / "optomerge"
if _pkg.is_dir() and str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from optomerge.io          import load_frames
from optomerge.processing  import norm_image, subtract_background, crop_channel
from optomerge.registration import calculate_alignment
from optomerge.segmentation import find_channel_bounds
from optomerge.transform   import transform_image, zero_pad_images
from scipy.ndimage         import median_filter

# ── timer ─────────────────────────────────────────────────────────────────

_log: list[tuple[str, float]] = []

@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    _log.append((label, dt))
    print(f"  {label:<45s} {dt:6.2f}s")


# ── main profiling run ─────────────────────────────────────────────────────

def profile(src: Path, max_frames: int | None):
    print(f"\nProfiling: {src.name}")
    print("=" * 60)

    # 1. Load ----------------------------------------------------------------
    with timed("1. load_frames"):
        mov = load_frames(src, frame_max=max_frames)   # (H, W, N)
    H, W, N = mov.shape
    print(f"     → {H}×{W} px, {N} frames")

    # 2. Project -------------------------------------------------------------
    with timed("2. z_project (max + median filter)"):
        max_proj  = mov.max(axis=2)
        max_proj  = median_filter(max_proj, size=3)
        mean_proj = mov.mean(axis=2)

    # 3. Find channel bounds -------------------------------------------------
    with timed("3. find_channel_bounds (k-means + line search)"):
        b1, s1, b2, s2 = find_channel_bounds(
            max_proj, mean_projection=mean_proj, verbose=False
        )
    two_ch = b2 is not None
    print(f"     → {'two channels' if two_ch else 'one channel'}")

    # 4. Calculate alignment -------------------------------------------------
    with timed("4. calculate_alignment (scale/rot FFT search)"):
        if two_ch:
            g_crop = crop_channel(mean_proj, b1, s1)
            r_crop = crop_channel(mean_proj, b2, s2)
            g_min, g_max = g_crop.min(), g_crop.max()
            r_min, r_max = r_crop.min(), r_crop.max()
            g_n = norm_image(g_crop, g_min, g_max)
            r_n = norm_image(r_crop, r_min, r_max)
            t1, t2, rot, sc1, sc2, score = calculate_alignment(g_n, r_n)
        else:
            g_min, g_max = mov.min(), mov.max()
            r_min, r_max = g_min, g_max
            t1=t2=rot=0.0; sc1=sc2=1.0; score=0.0

    # 5. align_frames — decomposed into sub-steps ----------------------------
    print("\n  align_frames breakdown:")

    with timed("5a.  crop_channel (green, all frames)"):
        green = crop_channel(mov, b1, s1)

    with timed("5b.  norm_image (green, all frames)"):
        green = norm_image(green, g_min, g_max)

    with timed("5c.  subtract_background green (grey_opening × N)"):
        green = subtract_background(green, radius=10)

    if two_ch:
        with timed("5d.  crop_channel (red, all frames)"):
            red = crop_channel(mov, b2, s2)

        with timed("5e.  norm_image (red, all frames)"):
            red = norm_image(red, r_min, r_max)

        with timed("5f.  zero_pad_images"):
            red, green_pad = zero_pad_images(red, green)

        with timed("5g.  transform_image (red, all frames — spline)"):
            red = transform_image(red, rot, sc1, sc2, t1, t2)

        with timed("5h.  subtract_background red (grey_opening × N)"):
            red = subtract_background(red, radius=10)
    else:
        green_pad = green

    with timed("5i.  final crop (find non-zero overlap region)"):
        mean_g = green_pad.mean(axis=2) if two_ch else green.mean(axis=2)
        cr = mean_g > 0.01
        rows, cols = cr.any(axis=1), cr.any(axis=0)
        rb = int(np.argmax(rows));  re = int(len(rows)-1-np.argmax(rows[::-1]))
        cb = int(np.argmax(cols));  ce = int(len(cols)-1-np.argmax(cols[::-1]))

    with timed("5j.  assemble RGB array"):
        if two_ch:
            r_c = red  [rb:re+1, cb:ce+1, :]
            g_c = green_pad[rb:re+1, cb:ce+1, :]
        else:
            g_c = green[rb:re+1, cb:ce+1, :]
            r_c = np.zeros_like(g_c)
        Hout, Wout = g_c.shape[:2]
        rgb = np.empty((Hout, Wout, 3, N), dtype=np.float64)
        rgb[:,:,0,:] = r_c
        rgb[:,:,1,:] = g_c
        rgb[:,:,2,:] = 0.0
    print(f"     → output shape: {Hout}×{Wout}×3×{N}")

    # ── Summary ──────────────────────────────────────────────────────────────
    total = sum(dt for _, dt in _log)
    print("\n" + "=" * 60)
    print(f"{'STEP':<46} {'TIME':>6}  {'%':>5}")
    print("-" * 60)
    for label, dt in sorted(_log, key=lambda x: -x[1]):
        print(f"  {label:<44} {dt:6.2f}s  {100*dt/total:5.1f}%")
    print("-" * 60)
    print(f"  {'TOTAL':<44} {total:6.2f}s  100.0%")
    print("=" * 60)

    # Per-frame costs for the expensive steps
    print(f"\nPer-frame costs  ({N} frames, {H}×{W} px raw):")
    for label, dt in _log:
        if any(x in label for x in ["subtract_background","transform_image"]):
            print(f"  {label:<44} {1000*dt/N:6.1f} ms/frame")


def main():
    ap = argparse.ArgumentParser(description="Profile the optomerge pipeline.")
    ap.add_argument("file", nargs="?",
                    default=None,
                    help="Path to a TIFF movie file")
    ap.add_argument("--frames", type=int, default=None,
                    help="Max frames to load (default: all)")
    args = ap.parse_args()

    if args.file is None:
        # Auto-pick the first .tif found in test_data/
        root = Path(__file__).parent / "test_data"
        candidates = sorted(root.rglob("*.tif"))
        if not candidates:
            sys.exit("No .tif files found under test_data/")
        src = candidates[0]
        print(f"No file specified — using: {src}")
    else:
        src = Path(args.file)

    if not src.exists():
        sys.exit(f"File not found: {src}")

    profile(src, args.frames)


if __name__ == "__main__":
    main()
