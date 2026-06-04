#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pipeline.py
================
A step-by-step diagnostic test script for the optomerge alignment pipeline,
written in the ``Movie`` / ``Channel`` class style of the original 2018
``optomerge.py`` prototype.

For each TIFF movie found under ``test_data/`` (and sub-directories) the
script runs every pipeline stage in sequence and saves a set of diagnostic
PNG images alongside the aligned TIFF output, so you can visually inspect
what happened at each step:

    output_temp/
    └── <experiment>/
        └── <channel>/
            ├── <movie>_aligned.tif           ← final aligned RGB stack
            ├── <movie>_diag_01_projection.png ← max-intensity projection
            ├── <movie>_diag_02_segmentation.png ← channel boundaries on projection
            ├── <movie>_diag_03_channels.png   ← cropped channel pair (frame 0)
            ├── <movie>_diag_04_alignment.png  ← before/after overlay
            └── <movie>_diag_05_rgb_frame.png  ← first frame of aligned output

Usage
-----
Basic run (auto channel detection, process all test_data movies):

    python test_pipeline.py

Specify a different input or output directory:

    python test_pipeline.py --input test_data --output output_temp

Force a channel order instead of auto-detecting it:

    python test_pipeline.py --channel-order top_green_fils_bottom_red_heads

Only use the first 200 frames for the projection used in alignment detection
(much faster for long movies; alignment quality is usually the same):

    python test_pipeline.py --frames 200

Process a single file directly:

    python test_pipeline.py --file "test_data/MyLOVChar4/ch3/movie 01.tif"

Dry-run (list files without processing):

    python test_pipeline.py --dry-run

Show verbose per-step output:

    python test_pipeline.py --verbose

Skip saving diagnostic PNGs (only write the aligned TIFF):

    python test_pipeline.py --no-diag

Requirements
------------
The optomerge package and its dependencies must be installed:

    pip install -e optomerge/

    # or, if not yet installed, ensure optomerge/ is in the same directory
    # as this script — it will be found automatically.

Dependencies: numpy, scipy, scikit-image, tifffile, matplotlib
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# ── matplotlib: use non-interactive backend when running from a terminal ──
import matplotlib
matplotlib.use("Agg")          # write PNGs without needing a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Locate the optomerge package ──────────────────────────────────────────
_here = Path(__file__).resolve().parent
_pkg  = _here / "optomerge"
if _pkg.is_dir() and str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

try:
    from optomerge             import AlignmentPipeline
    from optomerge.io          import save_rgb_tiff
    from optomerge.processing  import norm_image, subtract_background, crop_channel
    from optomerge.registration import calculate_alignment
    from optomerge.segmentation import find_channel_bounds
    from optomerge.transform   import (
        transform_image,
        zero_pad_images,
        fft_shift_image,
        integer_shift_image,
    )
except ImportError as exc:
    sys.exit(
        f"Cannot import optomerge ({exc}).\n"
        f"Install it with:  pip install -e optomerge/\n"
        f"or ensure the optomerge/ folder is next to this script."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Movie class  (mirrors the 2018 prototype, with all methods filled in)
# ─────────────────────────────────────────────────────────────────────────────

class Movie:
    """
    Represents one raw TIRF movie and drives its alignment pipeline.

    This class reproduces the architecture of the original ``optomerge.py``
    prototype (2018) with all stub methods completed, broken syntax fixed,
    and diagnostic image saving added.

    Attributes
    ----------
    filename : str
        Bare filename of the movie (e.g. ``'movie 01.tif'``).
    directory : str
        Directory that contains *filename*.
    dt : float
        Frame interval in seconds (default 1.0).
    dx : float
        Pixel size in nm (default 80.65).
    mov : np.ndarray or None
        Raw movie data, shape ``(N, H, W)`` as loaded from tifffile.
        Internally re-shaped to ``(H, W, N)`` after loading.
    channel_config : str
        Channel order; one of the strings in *channel_config_set* or
        ``'auto'``.
    channels : Channel or None
        Channel object populated by :meth:`find_channels_and_alignment`.
    alignment_params : dict or None
        ``{'t1', 't2', 'rot', 's1', 's2', 'score'}`` as returned by
        :func:`optomerge.registration.calculate_alignment`.
    rgb_movie : np.ndarray or None
        Aligned output, shape ``(H_out, W_out, 3, N)``.
    """

    def __init__(self):
        self.filename          = None
        self.directory         = None
        self.dt                = 1.0       # frame interval, seconds
        self.dx                = 80.65     # pixel size, nm

        self.mov               = None      # (H, W, N) float64 after load
        self.num_frames        = 0

        # projections (set by z_project)
        self.max_projection    = None      # (H, W)
        self.mean_projection   = None      # (H, W)

        self.channel_config    = 'auto'
        self.channel_config_set = [
            'auto',
            'top_green_fils_bottom_red_heads',
            'top_red_heads_bottom_green_fils',
        ]

        self.channels          = None      # Channel object, set after detection
        self.alignment_params  = None      # dict, set after registration
        self.rgb_movie         = None      # (H,W,3,N), set after align_frames
        self._crop_range       = None      # (rowbeg,rowend,colbeg,colend)

        # internal normalisation limits (set in find_channels_and_alignment)
        self._green_min        = 0.0
        self._green_max        = 1.0
        self._red_min          = 0.0
        self._red_max          = 1.0

        # pipeline settings
        self.bg_radius              = 10
        self.bg_method              = "morph_open"
        self.backend                = "auto"
        self.alignment_mode         = "robust"   # 'robust' or 'single'
        self.robust_chunk_size      = 400
        self.robust_min_candidates  = 10
        self.robust_max_trials      = 30
        self.align_method           = "auto"     # 'auto'|'affine'|'fft_shift'|'quick_and_dirty'
        self.projection_frames      = None       # None = use all frames

    # ── I/O ──────────────────────────────────────────────────────────────

    def load(self):
        """Load the movie from ``self.directory / self.filename``."""
        import tifffile

        fpath = Path(self.directory) / self.filename
        if not fpath.exists():
            raise FileNotFoundError(f"File not found: {fpath}")

        with tifffile.TiffFile(str(fpath)) as tif:
            arr = tif.asarray()   # (N, H, W) or (H, W) for tifffile

        # Normalise to (H, W, N) float64
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]      # single frame
        # tifffile returns (N, H, W) for multi-page TIFFs
        self.mov = np.moveaxis(arr, 0, -1).astype(np.float64)  # → (H, W, N)
        self.num_frames = self.mov.shape[2]
        print(f"  Loaded: {self.mov.shape[1]}×{self.mov.shape[0]} px, "
              f"{self.num_frames} frames")

    # ── Projection ───────────────────────────────────────────────────────

    def z_project(self, projection_method: str = 'max_intensity') -> np.ndarray:
        """Collapse the movie along the time axis.

        Applies a 3×3 median filter to the max-intensity projection to remove
        cosmic-ray spikes (matches the original prototype).

        Parameters
        ----------
        projection_method : ``'max_intensity'`` or ``'mean'``

        Returns
        -------
        np.ndarray, shape (H, W)
        """
        from scipy.ndimage import median_filter as _mf

        # Use only the first N projection frames if requested
        mov = self.mov
        if self.projection_frames is not None:
            n = min(self.projection_frames, mov.shape[2])
            mov = mov[:, :, :n]

        if projection_method == 'max_intensity':
            proj = mov.max(axis=2)
            proj = _mf(proj, size=3)   # denoise hot pixels (from prototype)
            self.max_projection = proj
        elif projection_method == 'mean':
            proj = mov.mean(axis=2)
            self.mean_projection = proj
        else:
            raise ValueError(f"Unknown projection method: {projection_method!r}")

        return proj

    # ── Channel detection + alignment ────────────────────────────────────

    def find_channels_and_alignment(self, verbose: bool = False):
        """Detect channel boundaries and compute alignment parameters.

        When ``self.alignment_mode == 'robust'`` and the movie has at least
        ``self.robust_chunk_size`` frames, delegates to the multi-chunk robust
        alignment (Python port of MATLAB ``findChannelsAndAlignment``).
        Otherwise uses a single projection of all frames.

        Populates:
        * ``self.channels``         – :class:`Channel` object with bounds/scrub
        * ``self.alignment_params`` – alignment dict
        * ``self._green_min/max``, ``self._red_min/max``
        """
        if self.max_projection is None:
            self.z_project('max_intensity')
        if self.mean_projection is None:
            self.z_project('mean')

        N = self.num_frames
        use_robust = (
            self.alignment_mode == "robust"
            and N >= self.robust_chunk_size
        )

        if use_robust:
            self._find_channels_and_alignment_robust(verbose=verbose)
        else:
            self._find_channels_and_alignment_single(verbose=verbose)

    def _find_channels_and_alignment_single(self, verbose: bool = False):
        """Single-projection channel detection + alignment."""
        chan = Channel()
        chan.max_projection  = self.max_projection
        chan.mean_projection = self.mean_projection
        chan.find_bounds(channel_order=self.channel_config, verbose=verbose)
        self.channels = chan

        if chan.bounds2 is None:
            self.alignment_params = dict(t1=0, t2=0, rot=0, s1=1, s2=1, score=0)
            return

        g_crop = crop_channel(self.mean_projection, chan.bounds1, chan.scrub1)
        r_crop = crop_channel(self.mean_projection, chan.bounds2, chan.scrub2)
        self._green_min, self._green_max = float(g_crop.min()), float(g_crop.max())
        self._red_min,   self._red_max   = float(r_crop.min()), float(r_crop.max())

        g_norm = norm_image(g_crop, self._green_min, self._green_max)
        r_norm = norm_image(r_crop, self._red_min,   self._red_max)

        t1, t2, rot, s1, s2, score = calculate_alignment(g_norm, r_norm)
        self.alignment_params = dict(t1=t1, t2=t2, rot=rot, s1=s1, s2=s2, score=score)

    def _find_channels_and_alignment_robust(self, verbose: bool = False):
        """Multi-chunk robust alignment: delegates to AlignmentPipeline."""
        from scipy.ndimage import median_filter as _mf

        chunk     = self.robust_chunk_size
        mov       = self.mov           # (H, W, N)
        H, W, N   = mov.shape

        # Acceptance thresholds (matching MATLAB defaults)
        size_cutoff_row = 0.70
        size_cutoff_col = 0.85
        max_trans       = 25.0
        max_rot_deg     = 2.0
        max_scale_pct   = 2.0

        # Score weights (matching MATLAB scoreChannelAlignment)
        w_size   = 1.0
        w_corr   = 3.0
        corr_ref = 0.1

        offset_step = max(1, chunk // 10)
        candidates  = []
        trials      = 0
        cycle       = 0

        while True:
            frame_offset = cycle * offset_step
            frame_start  = frame_offset

            while frame_start + chunk <= N:
                if trials >= self.robust_max_trials:
                    break
                trials += 1
                frame_end = frame_start + chunk
                sub = mov[:, :, frame_start:frame_end]

                max_proj  = _mf(sub.max(axis=2), size=3)
                mean_proj = sub.mean(axis=2)

                try:
                    b1, sc1, b2, sc2 = find_channel_bounds(
                        max_proj,
                        mean_projection=mean_proj,
                        channel_order=self.channel_config,
                        verbose=False,
                    )
                except Exception:
                    frame_start += chunk
                    continue

                if b2 is None:
                    frame_start += chunk
                    continue

                try:
                    g_crop = crop_channel(mean_proj, b1, sc1)
                    r_crop = crop_channel(mean_proj, b2, sc2)
                    g_lo, g_hi = float(g_crop.min()), float(g_crop.max())
                    r_lo, r_hi = float(r_crop.min()), float(r_crop.max())
                    g_norm = norm_image(g_crop, g_lo, g_hi)
                    r_norm = norm_image(r_crop, r_lo, r_hi)
                    t1, t2, rot, s1, s2, ph_score = calculate_alignment(g_norm, r_norm)
                except Exception:
                    frame_start += chunk
                    continue

                n_ch      = 2
                chan_h    = abs(int(b1[0, 1]) - int(b1[0, 0]))
                chan_w    = abs(int(b1[1, 1]) - int(b1[1, 0]))
                rot_deg   = abs(np.degrees(rot))
                scale_dev = max(abs(s1 - 1.0), abs(s2 - 1.0)) * 100.0

                size_ok  = (chan_h / (H / n_ch)) >= size_cutoff_row and (chan_w / W) >= size_cutoff_col
                trans_ok = abs(t1) <= max_trans and abs(t2) <= max_trans
                rot_ok   = rot_deg < max_rot_deg
                scale_ok = scale_dev < max_scale_pct

                if not (size_ok and trans_ok and rot_ok and scale_ok):
                    frame_start += chunk
                    continue

                geom_ch   = np.sqrt(chan_h * chan_w)
                geom_im   = np.sqrt(H * W)
                size_norm  = geom_ch / (geom_im * n_ch)
                corr_norm  = ph_score / corr_ref
                composite  = (w_size * size_norm + w_corr * corr_norm) / (w_size + w_corr)

                if not np.isnan(composite):
                    candidates.append(dict(
                        t1=t1, t2=t2, rot=rot, s1=s1, s2=s2,
                        score=ph_score, composite=composite,
                        bounds1=b1, scrub1=sc1, bounds2=b2, scrub2=sc2,
                        mean_proj=mean_proj,
                        green_min=g_lo, green_max=g_hi,
                        red_min=r_lo,   red_max=r_hi,
                    ))
                    if verbose:
                        print(f"  Chunk [{frame_start}–{frame_end}] accepted: "
                              f"rot={np.degrees(rot):.3f}°  score={ph_score:.4f}  "
                              f"composite={composite:.4f}")

                frame_start += chunk

            if len(candidates) >= self.robust_min_candidates:
                break
            if trials >= self.robust_max_trials:
                break
            cycle += 1

        # Fall back to single if no valid candidates
        if not candidates:
            if verbose:
                print("  Robust: no valid candidates; falling back to single projection.")
            self._find_channels_and_alignment_single(verbose=verbose)
            return

        best = max(candidates, key=lambda c: c["composite"])
        n_cand = len(candidates)

        # Populate channel object from best candidate
        chan = Channel()
        chan.bounds1 = best["bounds1"]
        chan.scrub1  = best["scrub1"]
        chan.bounds2 = best["bounds2"]
        chan.scrub2  = best["scrub2"]
        chan.max_projection  = self.max_projection
        chan.mean_projection = best["mean_proj"]
        self.channels       = chan

        self._green_min = best["green_min"]
        self._green_max = best["green_max"]
        self._red_min   = best["red_min"]
        self._red_max   = best["red_max"]
        self.alignment_params = dict(
            t1=best["t1"], t2=best["t2"], rot=best["rot"],
            s1=best["s1"], s2=best["s2"], score=best["score"],
        )
        if verbose:
            print(f"  Robust: selected best of {n_cand} candidates  "
                  f"composite={best['composite']:.4f}")

    # ── Frame alignment ──────────────────────────────────────────────────

    def align_frames(self, verbose: bool = False) -> np.ndarray:
        """Apply the alignment to every frame and assemble an RGB movie.

        Equivalent to ``alignFramesWithRed`` / ``alignFramesNoRed`` in MATLAB.
        Respects ``self.align_method`` (``'auto'``, ``'affine'``,
        ``'fft_shift'``, or ``'quick_and_dirty'``).

        Returns
        -------
        np.ndarray, shape (H_out, W_out, 3, N)
        """
        ap = self.alignment_params
        ch = self.channels
        mov = self.mov
        N   = self.num_frames

        if ch.bounds2 is None:
            # ── Single channel ──
            green = norm_image(
                crop_channel(mov, ch.bounds1, ch.scrub1),
                self._green_min, self._green_max,
            )
            green = subtract_background(
                green, radius=self.bg_radius,
                method=self.bg_method, backend=self.backend,
            )
            mean_g = green.mean(axis=2)
            cr = mean_g > 0.01
            rows, cols = cr.any(axis=1), cr.any(axis=0)
            rb, re = int(np.argmax(rows)), int(len(rows)-1-np.argmax(rows[::-1]))
            cb, ce = int(np.argmax(cols)), int(len(cols)-1-np.argmax(cols[::-1]))
            self._crop_range = (rb, re, cb, ce)
            g_crop = green[rb:re+1, cb:ce+1, :]
            H_out, W_out = g_crop.shape[:2]
            rgb = np.zeros((H_out, W_out, 3, N))
            rgb[:, :, 1, :] = g_crop
            self.rgb_movie = rgb
            return rgb

        # ── Two channels ──
        # Determine which transform method to use
        method = self._resolve_align_method()
        if verbose:
            print(f"  Aligning — t=({ap['t1']:.3f},{ap['t2']:.3f}), "
                  f"rot={np.degrees(ap['rot']):.4f}°, "
                  f"s=({ap['s1']:.5f},{ap['s2']:.5f}), method={method}")

        green = norm_image(
            crop_channel(mov, ch.bounds1, ch.scrub1),
            self._green_min, self._green_max,
        )
        green = subtract_background(
            green, radius=self.bg_radius,
            method=self.bg_method, backend=self.backend,
        )

        # Subtract red background BEFORE zero-padding (smaller frame → faster)
        red = norm_image(
            crop_channel(mov, ch.bounds2, ch.scrub2),
            self._red_min, self._red_max,
        )
        red = subtract_background(
            red, radius=self.bg_radius,
            method=self.bg_method, backend=self.backend,
        )

        red, green = zero_pad_images(red, green)

        # Apply chosen alignment method
        if method == "quick_and_dirty":
            red = integer_shift_image(red, ap['t2'], ap['t1'])
        elif method == "fft_shift":
            red = fft_shift_image(red, ap['t2'], ap['t1'])
        else:
            red = transform_image(
                red, ap['rot'], ap['s1'], ap['s2'], ap['t1'], ap['t2'],
                backend=self.backend,
            )

        if self._crop_range is None:
            mean_r, mean_g = red.mean(axis=2), green.mean(axis=2)
            cr = (mean_r > 0.01) & (mean_g > 0.01)
            rows, cols = cr.any(axis=1), cr.any(axis=0)
            rb = int(np.argmax(rows))
            re = int(len(rows) - 1 - np.argmax(rows[::-1]))
            cb = int(np.argmax(cols))
            ce = int(len(cols) - 1 - np.argmax(cols[::-1]))
            self._crop_range = (rb, re, cb, ce)
        rb, re, cb, ce = self._crop_range

        r_crop = red  [rb:re+1, cb:ce+1, :]
        g_crop = green[rb:re+1, cb:ce+1, :]
        H_out, W_out = r_crop.shape[:2]

        rgb = np.zeros((H_out, W_out, 3, N))
        rgb[:, :, 0, :] = r_crop
        rgb[:, :, 1, :] = g_crop
        self.rgb_movie = rgb
        return rgb

    def _resolve_align_method(self) -> str:
        """Return the concrete method string given ``self.align_method``.

        When ``'auto'``: uses ``'fft_shift'`` if rotation is < 0.5° **and**
        scale deviation is < 0.5 %; otherwise ``'affine'``.
        """
        method = self.align_method
        if method != "auto":
            return method
        ap = self.alignment_params
        rot_deg   = abs(np.degrees(ap['rot']))
        scale_dev = max(abs(ap['s1'] - 1.0), abs(ap['s2'] - 1.0))
        if rot_deg < 0.5 and scale_dev < 0.005:
            return "fft_shift"
        return "affine"

    # ── Save ─────────────────────────────────────────────────────────────

    def save_RGB(self, output_path: Path):
        """Save ``self.rgb_movie`` as a multi-page colour TIFF."""
        if self.rgb_movie is None:
            raise RuntimeError("Call align_frames() before save_RGB().")
        save_rgb_tiff(output_path, self.rgb_movie)
        print(f"  Saved: {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Channel class  (mirrors the 2018 prototype, find_bounds completed)
# ─────────────────────────────────────────────────────────────────────────────

class Channel:
    """Holds projections and detected bounds for the imaging channels.

    Mirrors the 2018 ``Channel`` class; ``find_bounds`` is now fully
    implemented using :func:`optomerge.segmentation.find_channel_bounds`.
    """

    def __init__(self):
        self.max_projection  = None   # (H, W)
        self.mean_projection = None   # (H, W)

        # Set after find_bounds()
        self.bounds1 = None   # np.ndarray (2,2) [[row_s,row_e],[col_s,col_e]]
        self.scrub1  = None   # (H,W) bool — pixels outside channel 1
        self.bounds2 = None   # same for channel 2, or None
        self.scrub2  = None

    def find_bounds(self, channel_order: str = 'auto', verbose: bool = False):
        """Detect channel boundaries using k-means segmentation + line search.

        Fills ``self.bounds1/2`` and ``self.scrub1/2``.

        The original 2018 prototype had the k-means segmentation in this
        method but left the line-search commented out; this version calls
        the completed implementation in ``optomerge.segmentation``.
        """
        b1, s1, b2, s2 = find_channel_bounds(
            self.max_projection,
            mean_projection=self.mean_projection,
            channel_order=channel_order,
            verbose=verbose,
        )
        self.bounds1, self.scrub1 = b1, s1
        self.bounds2, self.scrub2 = b2, s2

        if verbose:
            n_ch = 1 if b2 is None else 2
            print(f"  {n_ch} channel(s) detected.")
            print(f"  Ch1 rows {b1[0,0]}–{b1[0,1]}, "
                  f"cols {b1[1,0]}–{b1[1,1]}")
            if b2 is not None:
                print(f"  Ch2 rows {b2[0,0]}–{b2[0,1]}, "
                      f"cols {b2[1,0]}–{b2[1,1]}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic image saving
# ─────────────────────────────────────────────────────────────────────────────

def _disp_im(ax, im, title, cmap='hot', vmin=None, vmax=None):
    """Helper: show a 2-D image on *ax* with a title."""
    ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
              interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def save_diag_projection(movie: Movie, out_dir: Path, stem: str):
    """Diag 01 – max-intensity projection."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    _disp_im(ax, movie.max_projection,
             f"Max projection\n{stem}", cmap='gray')
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_diag_01_projection.png", dpi=150)
    plt.close(fig)


def save_diag_segmentation(movie: Movie, out_dir: Path, stem: str):
    """Diag 02 – channel boundaries overlaid on the projection."""
    ch = movie.channels
    proj = movie.max_projection.copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: raw projection
    _disp_im(axes[0], proj, "Max projection", cmap='gray')

    # Right: projection with boundary overlay
    vmin, vmax = np.percentile(proj, [1, 99])
    axes[1].imshow(proj, cmap='gray', vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='nearest')
    axes[1].set_title("Detected channels", fontsize=9)
    axes[1].axis('off')

    colours = ['lime', 'red']
    labels  = ['Channel 1 (green)', 'Channel 2 (red)']
    for i, (bounds, colour, label) in enumerate(
        zip([ch.bounds1, ch.bounds2], colours, labels)
    ):
        if bounds is None:
            continue
        r0, r1 = bounds[0, 0], bounds[0, 1]
        c0, c1 = bounds[1, 0], bounds[1, 1]
        rect = mpatches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            linewidth=2, edgecolor=colour, facecolor='none', label=label,
        )
        axes[1].add_patch(rect)

    axes[1].legend(loc='lower right', fontsize=7, framealpha=0.7)
    fig.suptitle(stem, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_diag_02_segmentation.png", dpi=150)
    plt.close(fig)


def save_diag_channels(movie: Movie, out_dir: Path, stem: str):
    """Diag 03 – cropped channel pair from frame 0."""
    ch  = movie.channels
    mov = movie.mov
    frame0 = mov[:, :, 0]
    ap = movie.alignment_params or {}

    g_crop = crop_channel(frame0, ch.bounds1, ch.scrub1)

    if ch.bounds2 is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        _disp_im(ax, g_crop, "Channel 1 (frame 0)", cmap='Greens_r')
    else:
        r_crop = crop_channel(frame0, ch.bounds2, ch.scrub2)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _disp_im(axes[0], g_crop, "Ch1 – green (frame 0)", cmap='Greens_r')
        _disp_im(axes[1], r_crop, "Ch2 – red (frame 0)",   cmap='Reds_r')

    fig.suptitle(stem, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_diag_03_channels.png", dpi=150)
    plt.close(fig)


def save_diag_alignment(movie: Movie, out_dir: Path, stem: str):
    """Diag 04 – before/after alignment overlay using mean projections."""
    ch = movie.channels
    ap = movie.alignment_params
    if ch.bounds2 is None or ap is None:
        return  # nothing to show for single-channel

    mean_proj = movie.mean_projection
    g_norm = norm_image(
        crop_channel(mean_proj, ch.bounds1, ch.scrub1),
        movie._green_min, movie._green_max,
    )
    r_norm = norm_image(
        crop_channel(mean_proj, ch.bounds2, ch.scrub2),
        movie._red_min, movie._red_max,
    )

    r_pad, g_pad = zero_pad_images(r_norm, g_norm)

    # Before: raw overlay
    H, W = r_pad.shape
    before = np.zeros((H, W, 3))
    before[:, :, 0] = r_pad
    before[:, :, 1] = g_pad

    # After: apply alignment
    r_aligned = transform_image(r_pad, ap['rot'], ap['s1'], ap['s2'],
                                ap['t1'], ap['t2'])
    after = np.zeros((H, W, 3))
    after[:, :, 0] = r_aligned
    after[:, :, 1] = g_pad

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.clip(before, 0, 1), aspect='auto')
    axes[0].set_title("Before alignment\n(red=ch2, green=ch1)", fontsize=9)
    axes[0].axis('off')
    axes[1].imshow(np.clip(after, 0, 1), aspect='auto')
    axes[1].set_title(
        f"After alignment\nt=({ap['t1']:.2f},{ap['t2']:.2f})  "
        f"rot={np.degrees(ap['rot']):.3f}°  "
        f"s=({ap['s1']:.4f},{ap['s2']:.4f})",
        fontsize=9,
    )
    axes[1].axis('off')
    fig.suptitle(stem, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_diag_04_alignment.png", dpi=150)
    plt.close(fig)


def save_diag_rgb_frame(movie: Movie, out_dir: Path, stem: str):
    """Diag 05 – first frame of the aligned RGB output."""
    if movie.rgb_movie is None:
        return
    rgb = movie.rgb_movie          # (H, W, 3, N)
    frame0 = np.clip(rgb[:, :, :, 0], 0, 1)   # (H, W, 3)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.imshow(frame0, aspect='auto')
    ax.set_title(f"Aligned RGB – frame 0\n{stem}", fontsize=9)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_diag_05_rgb_frame.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-file pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_file(
    src: Path,
    dst_dir: Path,
    channel_order: str,
    projection_frames: Optional[int],
    bg_radius: int,
    bg_method: str,
    backend: str,
    alignment_mode: str,
    robust_chunk_size: int,
    robust_min_candidates: int,
    robust_max_trials: int,
    align_method: str,
    save_diag: bool,
    verbose: bool,
) -> dict:
    """Run the full pipeline on *src* and save outputs to *dst_dir*."""
    result = dict(src=str(src), success=False, duration=0.0,
                  alignment=None, n_frames=None, error=None)
    t0 = time.perf_counter()

    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        stem = src.stem

        # ── 1. Instantiate Movie ──────────────────────────────────────────
        mov = Movie()
        mov.filename               = src.name
        mov.directory              = str(src.parent)
        mov.channel_config         = channel_order
        mov.bg_radius              = bg_radius
        mov.bg_method              = bg_method
        mov.backend                = backend
        mov.alignment_mode         = alignment_mode
        mov.robust_chunk_size      = robust_chunk_size
        mov.robust_min_candidates  = robust_min_candidates
        mov.robust_max_trials      = robust_max_trials
        mov.align_method           = align_method
        mov.projection_frames      = projection_frames

        # ── 2. Load ───────────────────────────────────────────────────────
        print("  [1/5] Loading ...")
        mov.load()
        result['n_frames'] = mov.num_frames

        # ── 3. Project ────────────────────────────────────────────────────
        print("  [2/5] Projecting ...")
        mov.z_project('max_intensity')
        mov.z_project('mean')

        if save_diag:
            save_diag_projection(mov, dst_dir, stem)

        # ── 4. Find channels + alignment ─────────────────────────────────
        print("  [3/5] Finding channels and computing alignment ...")
        mov.find_channels_and_alignment(verbose=verbose)

        ap = mov.alignment_params
        result['alignment'] = {k: (round(float(v), 5) if k != 'rot'
                                   else round(float(np.degrees(v)), 5))
                               for k, v in ap.items()}
        result['alignment']['rot_deg'] = result['alignment'].pop('rot')

        print(
            f"         t=({ap['t1']:.3f},{ap['t2']:.3f})  "
            f"rot={np.degrees(ap['rot']):.4f}°  "
            f"s=({ap['s1']:.5f},{ap['s2']:.5f})  "
            f"score={ap['score']:.4f}"
        )

        if save_diag:
            save_diag_segmentation(mov, dst_dir, stem)
            save_diag_channels(mov, dst_dir, stem)
            save_diag_alignment(mov, dst_dir, stem)

        # ── 5. Align frames ───────────────────────────────────────────────
        print("  [4/5] Aligning frames ...")
        mov.align_frames(verbose=verbose)

        if save_diag:
            save_diag_rgb_frame(mov, dst_dir, stem)

        # ── 6. Save ───────────────────────────────────────────────────────
        print("  [5/5] Saving ...")
        out_tif = dst_dir / f"{stem}_aligned.tif"
        mov.save_RGB(out_tif)
        print(f"  Output folder: {dst_dir}")

        result['success'] = True

    except Exception:
        result['error'] = traceback.format_exc()
        print(f"  FAILED:\n{result['error']}")

    result['duration'] = time.perf_counter() - t0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Step-by-step diagnostic test for the optomerge pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",  default="test_data",   metavar="DIR",
                   help="Root directory of raw movies  (default: test_data)")
    p.add_argument("--output", default="output_temp", metavar="DIR",
                   help="Root directory for output      (default: output_temp)")
    p.add_argument("--file",   default=None,           metavar="FILE",
                   help="Process a single file instead of scanning --input")
    p.add_argument("--channel-order", default="auto",
                   choices=["auto",
                            "top_green_fils_bottom_red_heads",
                            "top_red_heads_bottom_green_fils"],
                   metavar="STR",
                   help="Channel order (default: auto)")
    p.add_argument("--frames", type=int, default=None, metavar="N",
                   help="Max frames for projection (default: all)")
    p.add_argument("--bg-radius", type=int, default=10, metavar="N",
                   help="Background subtraction radius (default: 10)")
    p.add_argument("--bg-method", default="morph_open",
                   choices=["morph_open", "gaussian"],
                   dest="bg_method", metavar="STR",
                   help="Background method: morph_open (default) or gaussian")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "cv2", "scipy"],
                   metavar="STR",
                   help="Backend: auto (default), cv2, or scipy")
    p.add_argument("--alignment-mode", default="robust",
                   choices=["robust", "single"],
                   dest="alignment_mode", metavar="STR",
                   help="Alignment strategy: robust (default) or single")
    p.add_argument("--robust-chunk-size", type=int, default=400, metavar="N",
                   dest="robust_chunk_size",
                   help="Frames per chunk for robust alignment (default: 400)")
    p.add_argument("--robust-min-candidates", type=int, default=10, metavar="N",
                   dest="robust_min_candidates",
                   help="Min valid candidates for robust alignment (default: 10)")
    p.add_argument("--robust-max-trials", type=int, default=30, metavar="N",
                   dest="robust_max_trials",
                   help="Max chunks tried in robust alignment (default: 30)")
    p.add_argument("--align-method", default="auto",
                   choices=["auto", "affine", "fft_shift", "quick_and_dirty"],
                   dest="align_method", metavar="STR",
                   help="Alignment method: auto (default), affine, fft_shift, quick_and_dirty")
    p.add_argument("--no-diag", action="store_true",
                   help="Skip saving diagnostic PNG images")
    p.add_argument("--verbose", action="store_true",
                   help="Show detailed per-step progress")
    p.add_argument("--dry-run", action="store_true",
                   help="List files that would be processed, then exit")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    dst_root   = (script_dir / args.output).resolve()

    # ── Collect files ────────────────────────────────────────────────────
    if args.file:
        files = [Path(args.file).resolve()]
        src_root = files[0].parent
    else:
        src_root = (script_dir / args.input).resolve()
        if not src_root.is_dir():
            sys.exit(f"Input directory not found: {src_root}")
        files = sorted(
            p for p in src_root.rglob("*")
            if p.suffix.lower() in {".tif", ".tiff"} and p.is_file()
        )
        if not files:
            sys.exit(f"No .tif files found under {src_root}")

    # ── Dry run ──────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"Dry run — {len(files)} file(s) found:\n")
        for f in files:
            try:
                rel = f.relative_to(src_root)
            except ValueError:
                rel = f.name
            dst = dst_root / rel.parent / rel.stem
            print(f"  {rel}")
            print(f"    → {dst.relative_to(dst_root.parent)}/\n")
        return

    # ── Process ──────────────────────────────────────────────────────────
    print("=" * 68)
    print("optomerge diagnostic test pipeline")
    print(f"  Input : {src_root}")
    print(f"  Output: {dst_root}")
    print(f"  Files : {len(files)}")
    print(f"  BG method     : {args.bg_method}  backend: {args.backend}")
    print(f"  Alignment mode: {args.alignment_mode}"
          + (f"  chunk={args.robust_chunk_size}  "
             f"min_cands={args.robust_min_candidates}  "
             f"max_trials={args.robust_max_trials}"
             if args.alignment_mode == "robust" else ""))
    print(f"  Align method  : {args.align_method}")
    print(f"  Diag images : {'no' if args.no_diag else 'yes'}")
    print("=" * 68)

    results = []
    t_total = time.perf_counter()

    for idx, src in enumerate(files, 1):
        try:
            rel = src.relative_to(src_root)
        except ValueError:
            rel = Path(src.name)

        dst_dir = dst_root / rel.parent / rel.stem
        print(f"\n[{idx}/{len(files)}] {rel}")

        result = process_file(
            src=src,
            dst_dir=dst_dir,
            channel_order=args.channel_order,
            projection_frames=args.frames,
            bg_radius=args.bg_radius,
            bg_method=args.bg_method,
            backend=args.backend,
            alignment_mode=args.alignment_mode,
            robust_chunk_size=args.robust_chunk_size,
            robust_min_candidates=args.robust_min_candidates,
            robust_max_trials=args.robust_max_trials,
            align_method=args.align_method,
            save_diag=not args.no_diag,
            verbose=args.verbose,
        )
        results.append(result)

        dt = result['duration']
        status = "OK" if result['success'] else "FAILED"
        mins, secs = divmod(int(dt), 60)
        print(f"  [{status}] {mins}m {secs:02d}s")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    n_ok    = sum(1 for r in results if r['success'])
    n_fail  = len(results) - n_ok

    print("\n" + "=" * 68)
    print(f"DONE  {n_ok}/{len(results)} succeeded  "
          f"({int(elapsed//60)}m {int(elapsed%60):02d}s total)")

    if n_fail:
        print("\nFailed files:")
        for r in results:
            if not r['success']:
                print(f"  • {r['src']}")

    if n_ok:
        print("\nAlignment summary:")
        for r in results:
            if r['success'] and r['alignment']:
                a = r['alignment']
                name = Path(r['src']).name
                print(
                    f"  {name[:50]:<50}  "
                    f"t=({a.get('t1',0):.2f},{a.get('t2',0):.2f})  "
                    f"rot={a.get('rot_deg',0):.3f}°  "
                    f"score={a.get('score',0):.3f}"
                )
    print("=" * 68)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
