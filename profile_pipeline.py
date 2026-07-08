#!/usr/bin/env python3
"""
profile_pipeline.py
===================
Time every stage of the optomerge pipeline on one real movie and print a ranked
breakdown of where the time goes. Drives the object-oriented API stage by stage
(``RawMovie`` -> projections -> ``ChannelLayout.resolve`` -> ``Aligner.align``
-> ``RGBMovie.from_channels``).

Pipeline parameters come from built-in defaults, optional ``--config`` TOML
file(s), then explicit CLI flags (each layer overriding the previous). The
positional movie path is script-only.

Usage
-----
    python profile_pipeline.py "test_data/MyLOVChar4/ch3/movie 01.tif"
    python profile_pipeline.py "test_data/.../movie 01.tif" --frames 100
    python profile_pipeline.py "movie 01.tif" --config my_run.toml

The script writes no output files — it only measures time.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_here = Path(__file__).resolve().parent

from optomerge import RawMovie, RGBMovie, Settings

# argparse dests -> flat Settings fields (all default to argparse.SUPPRESS so
# only explicitly-passed flags override the config file). `file` is script-only.
_CLI_TO_FIELD = {
    "channel_order": "channel_order", "frames": "projection_frames",
    "bg_radius": "bg_radius", "use_scrub": "use_scrub", "upscale": "upscale",
    "verbose": "verbose",
}

_CHANNEL_ORDERS = (
    "auto",
    "top_green_fils_bottom_red_heads",
    "top_red_heads_bottom_green_fils",
)

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

def profile(src: Path, settings: Settings):
    max_frames = settings.channels.projection_frames
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
        layout = settings.build_layout().resolve(max_proj, mean_proj, verbose=False)
    two_ch = len(layout.specs) > 1
    print(f"     -> {'two channels' if two_ch else 'one channel'}")

    # 4. Calibrate channels + fit the alignment transform ------------------- #
    aligner = settings.build_aligner()
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
        rgb = RGBMovie.from_channels(channels, transforms, bg_radius=settings.processing.bg_radius)
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
    S = argparse.SUPPRESS
    ap = argparse.ArgumentParser(description="Profile the optomerge pipeline (OOP API).")
    ap.add_argument("file", nargs="?", default=None, help="Path to a TIFF movie file")
    ap.add_argument("--config", nargs="+", default=None, metavar="FILE",
                    help="TOML config file(s); later files and CLI flags override earlier ones")
    ap.add_argument("--channel-order", default=S, choices=_CHANNEL_ORDERS,
                    dest="channel_order", metavar="STR", help="Channel order (default: auto)")
    ap.add_argument("--frames", type=int, default=S,
                    help="Max frames used for projections/alignment (default: all)")
    ap.add_argument("--bg-radius", type=int, default=S, metavar="N", dest="bg_radius",
                    help="Background subtraction radius (default: 10)")
    ap.add_argument("--use-scrub", action="store_true", default=S, dest="use_scrub",
                    help="Align on upsampled scrub images for sub-pixel accuracy")
    ap.add_argument("--upscale", type=int, default=S, metavar="N",
                    help="Scrub upscaling factor when --use-scrub (default: 4)")
    args = ap.parse_args()

    # Resolution order: built-in defaults -> config file(s) -> explicit CLI flags.
    try:
        settings = Settings.from_toml(*args.config) if args.config else Settings()
    except (OSError, KeyError, RuntimeError) as exc:
        sys.exit(f"ERROR: could not load config: {exc}")
    overrides = {_CLI_TO_FIELD[dest]: val for dest, val in vars(args).items()
                 if dest in _CLI_TO_FIELD}
    settings = settings.with_overrides(**overrides)

    if args.file is None:
        root = _here / settings.io.input
        candidates = sorted(root.rglob("*.tif"))
        if not candidates:
            sys.exit(f"No .tif files found under {root}/")
        src = candidates[0]
        print(f"No file specified — using: {src}")
    else:
        src = Path(args.file)

    if not src.exists():
        sys.exit(f"File not found: {src}")
    profile(src, settings)


if __name__ == "__main__":
    main()
