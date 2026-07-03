#!/usr/bin/env python3
"""
profile_pipeline.py
===================
Time every stage of the optomerge pipeline on one real movie and print a ranked
breakdown of where the time goes. Drives the object-oriented API stage by stage
(``RawMovie`` -> projections -> ``ChannelLayout.resolve`` -> ``Aligner.align``
-> ``RGBMovie.from_channels``).

Usage
-----
    python profile_pipeline.py "test_data/MyLOVChar4/ch3/movie 01.tif"
    python profile_pipeline.py "test_data/.../movie 01.tif" --frames 100

The script writes no output files — it only measures time.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_here = Path(__file__).resolve().parent
_pkg = _here / "optomerge"
if _pkg.is_dir() and str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from optomerge import ChannelLayout, PhaseCorrelationAligner, RawMovie, RGBMovie

# ── timer ───────────────────────────────────────────────────────────────────

_log: list[tuple[str, float]] = []


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    _log.append((label, dt))
    print(f"  {label:<48s} {dt:6.2f}s")


# ── main profiling run ───────────────────────────────────────────────────────

def profile(src: Path, max_frames: int | None):
    print(f"\nProfiling: {src.name}")
    print("=" * 64)

    # 1. Open + realise the movie ------------------------------------------- #
    with timed("1. RawMovie.open + load frames"):
        movie = RawMovie.open(src)
        data = movie.to_array()               # (H, W, N)
    H, W, N = data.shape
    print(f"     -> {H}x{W} px, {N} frames")

    # 2. Projections -------------------------------------------------------- #
    with timed("2. max/mean projection"):
        max_proj = movie.max_projection(max_frames)
        mean_proj = movie.mean_projection(max_frames)

    # 3. Resolve the channel layout (segmentation) -------------------------- #
    with timed("3. ChannelLayout.resolve (k-means + line search)"):
        layout = ChannelLayout.auto().resolve(max_proj, mean_proj, verbose=False)
    two_ch = len(layout.specs) > 1
    print(f"     -> {'two channels' if two_ch else 'one channel'}")

    # 4. Calibrate channels + fit the alignment transform ------------------- #
    aligner = PhaseCorrelationAligner()
    with timed("4. aligner.align (scale/rot FFT search)"):
        proj_channels = {c.name: c.calibrate() for c in layout.split(mean_proj)}
        limits = {n: (c.vmin, c.vmax) for n, c in proj_channels.items()}
        reference = next(c for c in proj_channels.values() if c.reference)
        transforms = {
            spec.name: aligner.align(reference, proj_channels[spec.name])
            for spec in layout.moving_specs
        }
    for name, t in transforms.items():
        print(f"     -> {name}: {t}")

    # 5. Assemble the aligned RGB movie ------------------------------------- #
    with timed("5. RGBMovie.from_channels (crop/norm/transform/bg/assemble)"):
        channels = layout.split(data)
        for ch in channels:
            ch.vmin, ch.vmax = limits[ch.name]
            if ch.name in transforms:
                ch.transform = transforms[ch.name]
        rgb = RGBMovie.from_channels(channels, transforms, bg_radius=10)
    out = rgb.to_array()
    print(f"     -> output shape: {out.shape}")

    # ── Summary ────────────────────────────────────────────────────────────
    total = sum(dt for _, dt in _log)
    print("\n" + "=" * 64)
    print(f"{'STEP':<50} {'TIME':>6}  {'%':>5}")
    print("-" * 64)
    for label, dt in sorted(_log, key=lambda x: -x[1]):
        print(f"  {label:<48} {dt:6.2f}s  {100 * dt / total:5.1f}%")
    print("-" * 64)
    print(f"  {'TOTAL':<48} {total:6.2f}s  100.0%")
    print("=" * 64)

    print(f"\nPer-frame cost  ({N} frames, {H}x{W} px raw):")
    for label, dt in _log:
        if any(x in label for x in ["from_channels", "load"]):
            print(f"  {label:<48} {1000 * dt / N:6.1f} ms/frame")


def main():
    ap = argparse.ArgumentParser(description="Profile the optomerge pipeline (OOP API).")
    ap.add_argument("file", nargs="?", default=None, help="Path to a TIFF movie file")
    ap.add_argument("--frames", type=int, default=None,
                    help="Max frames used for projections/alignment (default: all)")
    args = ap.parse_args()

    if args.file is None:
        root = _here / "test_data"
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
