"""
optomerge.pipeline
==================
High-level pipeline that ties together I/O, channel finding, registration,
and frame alignment.

Python equivalent of MATLAB ``alignFrames.m``, ``alignFramesWithRed.m``, and
``alignFramesNoRed.m``, wrapped in an object-oriented API.

Typical usage
-------------
.. code-block:: python

    from optomerge import AlignmentPipeline

    pipe = AlignmentPipeline("movie.tif")
    pipe.find_channels()          # auto-detect channel bounds
    pipe.compute_alignment()      # phase-correlation alignment on projections
    rgb = pipe.align_frames()     # apply transform to every frame
    pipe.save("output_rgb.tif")   # write result

    # Or equivalently in one call:
    rgb = AlignmentPipeline("movie.tif").run()
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .io import load_frames, load_movie, save_rgb_tiff
from .processing import norm_image, subtract_background, crop_channel
from ._progress import ProgressBar, step_line
from .registration import calculate_alignment
from .segmentation import find_channel_bounds
from .transform import (
    fft_shift_image,
    integer_shift_image,
    transform_image,
    zero_pad_images,
)

# Import config lazily to avoid a hard circular dependency — used only in
# AlignmentPipeline.__init__ as a convenience shortcut.
# from .config import OptomergeConfig   (imported inside __init__ at runtime)

# ---------------------------------------------------------------------------
# Alignment-method auto-selection
# ---------------------------------------------------------------------------

#: Module-level fallback thresholds — used when no config object is provided.
_AUTO_ROT_DEG_THRESH: float = 0.5    # degrees
_AUTO_SCALE_DEV_THRESH: float = 0.005  # 0.5 %


def _select_align_method(
    alignment: "AlignmentParams",
    method: str,
    rot_deg_thresh: float = _AUTO_ROT_DEG_THRESH,
    scale_dev_thresh: float = _AUTO_SCALE_DEV_THRESH,
) -> str:
    """Return the concrete method string to use for frame alignment.

    When *method* is ``'auto'`` the function inspects *alignment* and returns
    ``'fft_shift'`` when rotation is below *rot_deg_thresh* **and** scale
    deviation is below *scale_dev_thresh*; otherwise falls back to
    ``'affine'``.

    For any explicit choice (``'affine'``, ``'fft_shift'``, or
    ``'quick_and_dirty'``) the value is returned unchanged.

    Parameters
    ----------
    alignment : AlignmentParams
        Computed alignment parameters.
    method : str
        Requested method (``'auto'``, ``'affine'``, ``'fft_shift'``, or
        ``'quick_and_dirty'``).
    rot_deg_thresh : float
        Maximum rotation (degrees) for auto-selection of fft_shift.
        Default: module-level ``_AUTO_ROT_DEG_THRESH`` (0.5°).
    scale_dev_thresh : float
        Maximum |s−1| for auto-selection of fft_shift.
        Default: module-level ``_AUTO_SCALE_DEV_THRESH`` (0.005).
    """
    if method != "auto":
        return method
    rot_deg   = abs(np.degrees(alignment.rot))
    scale_dev = max(abs(alignment.s1 - 1.0), abs(alignment.s2 - 1.0))
    if rot_deg < rot_deg_thresh and scale_dev < scale_dev_thresh:
        return "fft_shift"
    return "affine"


# ---------------------------------------------------------------------------
# Alignment parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class SharedAlignment:
    """Geometric alignment parameters shared across a batch of movies.

    Contains the channel-boundary geometry and the inter-channel affine
    transform extracted from one reference movie.  All movies in the batch
    that share the same optical setup (same day, same microscope) can reuse
    these parameters, skipping the per-movie channel detection and alignment
    search and only re-computing per-movie intensity normalisation.

    Attributes
    ----------
    bounds1 : np.ndarray, shape (2, 2)
        Channel 1 (green) crop bounds ``[[row_start, row_end],
        [col_start, col_end]]``.
    scrub1 : np.ndarray, shape (H, W)
        Boolean mask — True where channel 1 pixels are outside the valid
        region and should be zeroed.
    bounds2 : np.ndarray or None
        Channel 2 (red) bounds, same format as *bounds1*.  ``None`` for
        single-channel data.
    scrub2 : np.ndarray or None
        Scrub mask for channel 2, or ``None``.
    alignment : AlignmentParams
        Inter-channel affine transform (translation, rotation, scale).
    reference_file : str or None
        Path of the movie this alignment was computed from.
    """
    bounds1:        np.ndarray
    scrub1:         np.ndarray
    bounds2:        Optional[np.ndarray]
    scrub2:         Optional[np.ndarray]
    alignment:      "AlignmentParams"
    reference_file: Optional[str] = None

    def __repr__(self) -> str:
        n_ch = 2 if self.bounds2 is not None else 1
        return (
            f"SharedAlignment({n_ch}ch, "
            f"ref={Path(self.reference_file).name if self.reference_file else 'None'}, "
            f"alignment={self.alignment})"
        )


@dataclass
class AlignmentParams:
    """Transformation parameters that map channel 2 onto channel 1.

    Attributes
    ----------
    t1 : float
        Row translation (pixels).
    t2 : float
        Column translation (pixels).
    rot : float
        Rotation angle (radians).
    s1 : float
        X-axis scale factor.
    s2 : float
        Y-axis scale factor.
    score : float
        Phase-correlation score of the best alignment found.
    """
    t1: float = 0.0
    t2: float = 0.0
    rot: float = 0.0
    s1: float = 1.0
    s2: float = 1.0
    score: float = 0.0

    def __repr__(self) -> str:
        return (
            f"AlignmentParams(t=({self.t1:.3f}, {self.t2:.3f}), "
            f"rot={self.rot:.5f} rad, s=({self.s1:.5f}, {self.s2:.5f}), "
            f"score={self.score:.4f})"
        )


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class AlignmentPipeline:
    """End-to-end pipeline for dual-channel fluorescence movie alignment.

    Parameters
    ----------
    filename : str or Path
        Path to the raw TIFF movie.
    channel_order : str, optional
        ``'auto'`` (default), ``'top_green_fils_bottom_red_heads'``, or
        ``'top_red_heads_bottom_green_fils'``.
    channel_bounds : array-like, shape (1,4) or (2,4), optional
        Manually specified channel bounds ``[x_left, y_top, x_right, y_bottom]``
        per channel.  When provided the automatic boundary search is skipped.
    projection_frames : int or None, optional
        Number of frames to use for the projection used in channel/alignment
        detection.  ``None`` (default) uses all frames.
    chunk_size : int, optional
        Frames to process at a time in :meth:`align_frames`.  Default 100.
    bg_radius : int, optional
        Radius of the background-subtraction structuring element.  Default 10.
    bg_method : str, optional
        Background subtraction method: ``'morph_open'`` (default, morphological
        opening) or ``'gaussian'`` (Gaussian blur, ~10–30× faster).
    n_workers : int or None, optional
        Number of CPU threads for parallel frame processing.  ``None`` (default)
        uses all logical CPUs.  Pass ``1`` to disable parallelism.
    backend : str, optional
        Computational backend for background subtraction and affine transform.
        ``'auto'`` (default) uses OpenCV when available, falling back to scipy.
        ``'cv2'`` forces OpenCV.  ``'scipy'`` forces scipy/skimage.
    alignment_mode : str, optional
        Strategy for computing the channel alignment:

        ``'robust'`` (default)
            Mimics MATLAB ``findChannelsAndAlignment``.  Splits the movie into
            chunks of ``robust_chunk_size`` frames, projects each chunk, finds
            channels and computes alignment for every chunk, then selects the
            candidate with the highest composite score (weighted channel size +
            phase-correlation score).  Falls back to ``'single'`` automatically
            when the movie is shorter than ``robust_chunk_size``.
        ``'single'``
            Classic single-projection mode: project all (or
            ``projection_frames``) frames, compute alignment once.
    robust_chunk_size : int, optional
        Frames per chunk when ``alignment_mode='robust'``.  Default 400.
    robust_min_candidates : int, optional
        Minimum valid candidates to collect before picking the best.
        Default 10.
    robust_max_trials : int, optional
        Hard limit on total chunks tried.  Default 30.
    shared_alignment : SharedAlignment or None, optional
        Pre-computed geometric alignment from a reference movie.  When
        provided, :meth:`find_channels` and :meth:`compute_alignment` skip
        their detection / search steps and load bounds and transform
        parameters directly from this object.  Intensity normalisation
        (``green_min/max``, ``red_min/max``) is still derived from the
        current movie's own projection, so per-movie fluorescence
        differences are handled correctly.  Default ``None`` (compute
        alignment from scratch for every movie).
    align_method : str, optional
        Frame-alignment method to apply after computing alignment parameters:

        ``'auto'`` (default)
            Uses ``'fft_shift'`` when rotation is < 0.5° **and** scale
            deviation is < 0.5 % for both axes; falls back to ``'affine'``
            otherwise.
        ``'affine'``
            Full affine transform (rotation + anisotropic scale + translation)
            via cubic-spline interpolation.  Most accurate, slowest.
        ``'fft_shift'``
            Sub-pixel translation via the Fourier shift theorem.  Exact for
            pure translations; ignores residual rotation / scale.  ~2–3×
            faster than affine.
        ``'quick_and_dirty'``
            Integer-pixel translation only (no interpolation).  Fastest
            (~20–50×), ±0.5 px accuracy.
    show_progress : bool, optional
        Display a live progress bar on stderr during slow steps.  Default True.
    verbose : bool, optional
        Print detailed alignment parameters.  Default False.
    config : OptomergeConfig or None, optional
        A fully resolved configuration object.  When provided, its values are
        used as defaults for every parameter that was not explicitly supplied
        to this constructor.  Explicit keyword arguments always take
        precedence over the config.  Useful when running from the CLI (the
        config is loaded once and passed to every pipeline) or from a
        notebook (load a shared config, override per-movie settings as
        needed).
    """

    def __init__(
        self,
        filename: str | Path,
        channel_order: Optional[str] = None,
        channel_bounds: Optional[np.ndarray] = None,
        projection_frames: Optional[int] = None,
        chunk_size: int = 100,
        bg_radius: Optional[int] = None,
        bg_method: Optional[str] = None,
        n_workers: Optional[int] = None,
        backend: Optional[str] = None,
        alignment_mode: Optional[str] = None,
        robust_chunk_size: Optional[int] = None,
        robust_min_candidates: Optional[int] = None,
        robust_max_trials: Optional[int] = None,
        shared_alignment: Optional["SharedAlignment"] = None,
        align_method: Optional[str] = None,
        show_progress: Optional[bool] = None,
        verbose: Optional[bool] = None,
        config: Optional["OptomergeConfig"] = None,  # type: ignore[name-defined]
    ) -> None:
        # ------------------------------------------------------------------
        # Resolve effective values: explicit kwarg > config > hard default
        # ------------------------------------------------------------------
        from .config import OptomergeConfig  # local import avoids circularity

        cfg = config if config is not None else OptomergeConfig()

        self.filename = Path(filename)
        self.channel_order = channel_order if channel_order is not None \
            else cfg.projection.channel_order
        self.channel_bounds_manual = (
            np.asarray(channel_bounds) if channel_bounds is not None else None
        )
        self.projection_frames = projection_frames if projection_frames is not None \
            else cfg.projection.frames
        self.chunk_size = chunk_size
        self.bg_radius = bg_radius if bg_radius is not None \
            else cfg.projection.bg_radius
        self.bg_method = bg_method if bg_method is not None \
            else cfg.projection.bg_method
        self.n_workers = n_workers if n_workers is not None \
            else cfg.compute.n_workers
        self.backend = backend if backend is not None \
            else cfg.compute.backend
        self.alignment_mode = alignment_mode if alignment_mode is not None \
            else cfg.alignment.mode
        self.robust_chunk_size = robust_chunk_size if robust_chunk_size is not None \
            else cfg.alignment.robust_chunk_size
        self.robust_min_candidates = robust_min_candidates if robust_min_candidates is not None \
            else cfg.alignment.robust_min_candidates
        self.robust_max_trials = robust_max_trials if robust_max_trials is not None \
            else cfg.alignment.robust_max_trials
        self.shared_alignment = shared_alignment
        self.align_method = align_method if align_method is not None \
            else cfg.frames.align_method
        self.show_progress = show_progress if show_progress is not None \
            else cfg.logging.show_progress
        self.verbose = verbose if verbose is not None \
            else cfg.logging.verbose

        # Auto-method thresholds — from config, not overridable per-instance
        # (change them in the config file instead).
        self._auto_rot_deg_thresh: float = (
            cfg.alignment.auto_method_thresholds.rotation_deg
        )
        self._auto_scale_dev_thresh: float = (
            cfg.alignment.auto_method_thresholds.scale_percent / 100.0
        )

        # ---- State set during pipeline stages ----
        self._movie: Optional[np.ndarray] = None  # (H, W, N) raw movie
        self._max_proj: Optional[np.ndarray] = None
        self._mean_proj: Optional[np.ndarray] = None

        # Channel 1 = green (bright), channel 2 = red (dim) / None for 1-ch
        self.bounds1: Optional[np.ndarray] = None   # (2,2)
        self.scrub1: Optional[np.ndarray] = None    # (H,W) bool
        self.bounds2: Optional[np.ndarray] = None
        self.scrub2: Optional[np.ndarray] = None

        self.alignment: Optional[AlignmentParams] = None

        # Intensity limits for normalisation (set in compute_alignment)
        self._green_min: float = 0.0
        self._green_max: float = 1.0
        self._red_min: float = 0.0
        self._red_max: float = 1.0

        # Output movie (H, W, 3, N) or (H, W, 1, N) for single-channel
        self._rgb_movie: Optional[np.ndarray] = None
        # Crop range determined during first chunk (reused for subsequent)
        self._crop_range: Optional[Tuple[int, int, int, int]] = None

    # ------------------------------------------------------------------ #
    #  Stage 1: load                                                       #
    # ------------------------------------------------------------------ #

    def load(
        self,
        frame_min: int = 0,
        frame_max: Optional[int] = None,
    ) -> "AlignmentPipeline":
        """Load the TIFF movie into memory.

        Parameters
        ----------
        frame_min, frame_max : int, optional
            Frame range to load.  Defaults to the full movie.

        Returns
        -------
        self (for chaining)
        """
        t0 = time.perf_counter()
        if self.verbose:
            print(f"Loading {self.filename} ...")
        self._movie = load_frames(self.filename, frame_min, frame_max)
        elapsed = time.perf_counter() - t0
        if self.show_progress:
            H, W, N = self._movie.shape
            step_line(f"Loading  ({N} frames, {H}×{W} px)", elapsed)
        elif self.verbose:
            H, W, N = self._movie.shape
            print(f"  Loaded {N} frames, size {H}×{W}.")
        return self

    def _ensure_loaded(self) -> None:
        if self._movie is None:
            self.load()

    # ------------------------------------------------------------------ #
    #  Stage 2: projection                                                 #
    # ------------------------------------------------------------------ #

    def project(self) -> "AlignmentPipeline":
        """Compute max-intensity and mean projections for use in channel finding.

        Returns
        -------
        self
        """
        self._ensure_loaded()
        t0 = time.perf_counter()
        mov = self._movie
        if self.projection_frames is not None:
            n = min(self.projection_frames, mov.shape[2])
            mov = mov[:, :, :n]

        self._max_proj  = mov.max(axis=2)
        self._mean_proj = mov.mean(axis=2)
        if self.show_progress:
            step_line("Projection  (max + mean)", time.perf_counter() - t0)
        return self

    # ------------------------------------------------------------------ #
    #  Stage 3: find channels                                              #
    # ------------------------------------------------------------------ #

    def find_channels(self) -> "AlignmentPipeline":
        """Locate channel boundaries in the projected image.

        When ``self.shared_alignment`` is set the bounds and scrub masks are
        loaded directly from the shared object — no detection is run.

        Equivalent to MATLAB ``findChannelBounds``.

        Returns
        -------
        self
        """
        if self._mean_proj is None:
            self.project()

        # ---- fast path: reuse bounds from shared alignment ----
        if self.shared_alignment is not None:
            sa = self.shared_alignment
            self.bounds1, self.scrub1 = sa.bounds1, sa.scrub1
            self.bounds2, self.scrub2 = sa.bounds2, sa.scrub2
            n_ch = 2 if self.bounds2 is not None else 1
            if self.show_progress:
                step_line(f"Channel detection  ({n_ch}ch, shared)")
            if self.verbose:
                print(f"  Using shared bounds from {sa.reference_file}")
            return self

        t0 = time.perf_counter()
        if self.verbose:
            print("Finding channel boundaries ...")

        b1, s1, b2, s2 = find_channel_bounds(
            self._max_proj,
            mean_projection=self._mean_proj,
            channel_order=self.channel_order,
            channel_bounds=self.channel_bounds_manual,
            verbose=self.verbose,
        )
        self.bounds1, self.scrub1 = b1, s1
        self.bounds2, self.scrub2 = b2, s2
        elapsed = time.perf_counter() - t0

        n_ch = 2 if self.bounds2 is not None else 1
        if self.show_progress:
            step_line(f"Channel detection  ({n_ch} channel{'s' if n_ch > 1 else ''})", elapsed)
        if self.verbose:
            print(f"  Channel 1 bounds: {self.bounds1}")
            if self.bounds2 is not None:
                print(f"  Channel 2 bounds: {self.bounds2}")
            else:
                print("  Single channel detected.")
        return self

    # ------------------------------------------------------------------ #
    #  Stage 4: compute alignment                                          #
    # ------------------------------------------------------------------ #

    def compute_alignment(
        self,
        init_rot: float = 0.0,
        init_s1: float = 1.0,
        init_s2: float = 1.0,
    ) -> "AlignmentPipeline":
        """Calculate the affine alignment between the two channels.

        Dispatches to :meth:`_compute_alignment_single` or
        :meth:`_compute_alignment_robust` depending on
        ``self.alignment_mode``.

        Parameters
        ----------
        init_rot, init_s1, init_s2 : float
            Initial guesses for rotation and scale parameters.

        Returns
        -------
        self
        """
        self._ensure_loaded()

        if self.bounds1 is None:
            self.find_channels()

        if self.bounds2 is None:
            # Single channel – identity alignment, no mode selection needed
            self.alignment = AlignmentParams()
            if self.show_progress:
                step_line("Alignment  (single channel)")
            return self

        N = self._movie.shape[2]
        use_robust = (
            self.alignment_mode == "robust"
            and N >= self.robust_chunk_size
        )

        if use_robust:
            return self._compute_alignment_robust(init_rot, init_s1, init_s2)
        else:
            return self._compute_alignment_single(init_rot, init_s1, init_s2)

    def _compute_alignment_single(
        self,
        init_rot: float = 0.0,
        init_s1: float = 1.0,
        init_s2: float = 1.0,
    ) -> "AlignmentPipeline":
        """Classic single-projection alignment.

        When ``self.shared_alignment`` is set the transform parameters are
        loaded from the shared object; only the per-movie intensity limits
        (``_green_min/max``, ``_red_min/max``) are (re-)computed from the
        current movie's mean projection.
        """
        t0 = time.perf_counter()

        green_proj = crop_channel(self._mean_proj, self.bounds1, self.scrub1)
        self._green_min = float(green_proj.min())
        self._green_max = float(green_proj.max())

        if self.bounds2 is not None:
            red_proj = crop_channel(self._mean_proj, self.bounds2, self.scrub2)
            self._red_min = float(red_proj.min())
            self._red_max = float(red_proj.max())

        # ---- fast path: reuse transform from shared alignment ----
        if self.shared_alignment is not None:
            sa = self.shared_alignment
            self.alignment = AlignmentParams(
                t1=sa.alignment.t1, t2=sa.alignment.t2,
                rot=sa.alignment.rot, s1=sa.alignment.s1,
                s2=sa.alignment.s2, score=sa.alignment.score,
            )
            elapsed = time.perf_counter() - t0
            if self.show_progress:
                a = self.alignment
                chosen = _select_align_method(
                    self.alignment, self.align_method,
                    rot_deg_thresh=self._auto_rot_deg_thresh,
                    scale_dev_thresh=self._auto_scale_dev_thresh,
                )
                step_line(
                    f"Alignment  (shared  rot={np.degrees(a.rot):.3f}°  →{chosen})",
                    elapsed,
                )
            if self.verbose:
                print(f"  Using shared alignment: {self.alignment}")
            return self

        # ---- normal path: compute alignment from scratch ----
        if self.verbose:
            print("Computing alignment (single projection) ...")
        if self.bounds2 is None:
            return self

        g_norm = norm_image(green_proj, self._green_min, self._green_max)
        r_norm = norm_image(
            crop_channel(self._mean_proj, self.bounds2, self.scrub2),
            self._red_min, self._red_max,
        )

        t1, t2, rot, s1, s2, score = calculate_alignment(
            g_norm, r_norm, init_rot, init_s1, init_s2
        )
        self.alignment = AlignmentParams(t1=t1, t2=t2, rot=rot,
                                         s1=s1, s2=s2, score=score)
        elapsed = time.perf_counter() - t0
        if self.show_progress:
            chosen = _select_align_method(
                    self.alignment, self.align_method,
                    rot_deg_thresh=self._auto_rot_deg_thresh,
                    scale_dev_thresh=self._auto_scale_dev_thresh,
                )
            step_line(
                f"Alignment  (rot={np.degrees(rot):.3f}°  "
                f"score={score:.3f}  →{chosen})",
                elapsed,
            )
        if self.verbose:
            print(f"  Alignment: {self.alignment}")
        return self

    def _compute_alignment_robust(
        self,
        init_rot: float = 0.0,
        init_s1: float = 1.0,
        init_s2: float = 1.0,
    ) -> "AlignmentPipeline":
        """Robust multi-chunk alignment: Python port of MATLAB
        ``findChannelsAndAlignment``.

        Splits the movie into non-overlapping chunks of
        ``self.robust_chunk_size`` frames.  For each chunk, projects the
        sub-stack, detects channels, and computes alignment.  Every candidate
        that passes acceptance-criteria filters is scored with a composite
        metric (weighted channel size + phase-correlation score).  When fewer
        than ``self.robust_min_candidates`` valid candidates have been found
        after one full pass, the start offset is shifted by 10 % of the chunk
        size and another pass is attempted, up to ``self.robust_max_trials``
        total chunks.  The candidate with the highest composite score is used.

        Acceptance criteria (matching MATLAB defaults)
        -----------------------------------------------
        * Channel height ≥ 70 % of (image height / n_channels)
        * Channel width  ≥ 85 % of image width
        * |t1|, |t2|     ≤ 25 px
        * |rot|           < 2°
        * |s − 1| × 100  < 2 %

        Composite score (MATLAB ``scoreChannelAlignment`` weights)
        ----------------------------------------------------------
        score = (1.0 × channel_size_norm + 3.0 × phase_corr_norm) / 4.0

        where ``phase_corr_norm = alignment.score / 0.1``.

        When ``self.shared_alignment`` is set the search is skipped entirely;
        only per-movie intensity limits are (re-)computed.
        """
        # ---- fast path: shared alignment supplied ----
        if self.shared_alignment is not None:
            return self._compute_alignment_single(init_rot, init_s1, init_s2)

        from scipy.ndimage import median_filter as _mf

        chunk  = self.robust_chunk_size
        mov    = self._movie           # (H, W, N)
        H, W, N = mov.shape

        # Acceptance thresholds (matching MATLAB)
        size_cutoff_row = 0.70   # channel height ≥ 70 % of H/n_ch
        size_cutoff_col = 0.85   # channel width  ≥ 85 % of W
        max_trans  = 25.0        # px
        max_rot_deg = 2.0        # degrees
        max_scale_pct = 2.0      # percent

        # Score weights
        w_size   = 1.0
        w_corr   = 3.0
        corr_ref = 0.1           # normalisation reference for phase-corr score

        # --- offset step: 10 % of chunk size, rounded to at least 1 ---
        offset_step = max(1, chunk // 10)

        candidates: list = []   # list of dicts
        trials = 0

        sp = self.show_progress
        if sp:
            step_line("Robust alignment  (scanning chunks)")

        cycle = 0
        while True:
            frame_offset = cycle * offset_step
            frame_start  = frame_offset

            while frame_start + chunk <= N:
                if trials >= self.robust_max_trials:
                    break
                trials += 1
                frame_end = frame_start + chunk

                # --- project this sub-stack ---
                sub = mov[:, :, frame_start:frame_end]
                max_proj  = _mf(sub.max(axis=2), size=3)
                mean_proj = sub.mean(axis=2)

                try:
                    b1, sc1, b2, sc2 = find_channel_bounds(
                        max_proj,
                        mean_projection=mean_proj,
                        channel_order=self.channel_order,
                        channel_bounds=self.channel_bounds_manual,
                        verbose=False,
                    )
                except Exception:
                    frame_start += chunk
                    continue

                if b2 is None:
                    frame_start += chunk
                    continue

                # --- compute alignment on this chunk's projection ---
                try:
                    g_crop = crop_channel(mean_proj, b1, sc1)
                    r_crop = crop_channel(mean_proj, b2, sc2)
                    g_lo, g_hi = float(g_crop.min()), float(g_crop.max())
                    r_lo, r_hi = float(r_crop.min()), float(r_crop.max())
                    g_norm = norm_image(g_crop, g_lo, g_hi)
                    r_norm = norm_image(r_crop, r_lo, r_hi)
                    t1, t2, rot, s1, s2, ph_score = calculate_alignment(
                        g_norm, r_norm, init_rot, init_s1, init_s2
                    )
                except Exception:
                    frame_start += chunk
                    continue

                # --- acceptance filter ---
                n_ch        = 2
                chan_h      = abs(int(b1[0, 1]) - int(b1[0, 0]))
                chan_w      = abs(int(b1[1, 1]) - int(b1[1, 0]))
                rot_deg     = abs(np.degrees(rot))
                scale_dev   = max(abs(s1 - 1.0), abs(s2 - 1.0)) * 100.0

                size_ok = (
                    (chan_h / (H / n_ch)) >= size_cutoff_row
                    and (chan_w / W)       >= size_cutoff_col
                )
                trans_ok  = abs(t1) <= max_trans and abs(t2) <= max_trans
                rot_ok    = rot_deg  < max_rot_deg
                scale_ok  = scale_dev < max_scale_pct

                if not (size_ok and trans_ok and rot_ok and scale_ok):
                    if self.verbose:
                        print(
                            f"  Chunk [{frame_start}–{frame_end}] rejected: "
                            f"size_ok={size_ok} trans_ok={trans_ok} "
                            f"rot_ok={rot_ok} scale_ok={scale_ok}"
                        )
                    frame_start += chunk
                    continue

                # --- composite score ---
                geom_ch  = np.sqrt(chan_h * chan_w)
                geom_im  = np.sqrt(H * W)
                size_norm = geom_ch / (geom_im * n_ch)
                corr_norm = ph_score / corr_ref
                composite = (w_size * size_norm + w_corr * corr_norm) / (
                    w_size + w_corr
                )

                if np.isnan(composite):
                    frame_start += chunk
                    continue

                candidates.append(dict(
                    t1=t1, t2=t2, rot=rot, s1=s1, s2=s2,
                    score=ph_score, composite=composite,
                    bounds1=b1, scrub1=sc1,
                    bounds2=b2, scrub2=sc2,
                    mean_proj=mean_proj,
                    green_min=g_lo, green_max=g_hi,
                    red_min=r_lo,   red_max=r_hi,
                    frame_start=frame_start, frame_end=frame_end,
                ))
                if self.verbose:
                    print(
                        f"  Chunk [{frame_start}–{frame_end}] accepted: "
                        f"rot={np.degrees(rot):.3f}°  "
                        f"score={ph_score:.4f}  composite={composite:.4f}"
                    )

                frame_start += chunk

            # ---- break conditions ----
            if len(candidates) >= self.robust_min_candidates:
                break
            if trials >= self.robust_max_trials:
                if self.verbose:
                    print(
                        f"  Robust alignment: max trials ({self.robust_max_trials}) "
                        "reached without enough candidates."
                    )
                break
            cycle += 1

        # ---- select best candidate or fall back to single ----
        if not candidates:
            if self.verbose:
                print("  No valid candidates found; falling back to single projection.")
            if sp:
                step_line("Alignment  (robust→fallback single)")
            return self._compute_alignment_single(init_rot, init_s1, init_s2)

        best = max(candidates, key=lambda c: c["composite"])

        # Update pipeline state from best candidate
        self.bounds1, self.scrub1 = best["bounds1"], best["scrub1"]
        self.bounds2, self.scrub2 = best["bounds2"], best["scrub2"]
        self._green_min = best["green_min"]
        self._green_max = best["green_max"]
        self._red_min   = best["red_min"]
        self._red_max   = best["red_max"]
        self.alignment  = AlignmentParams(
            t1=best["t1"], t2=best["t2"], rot=best["rot"],
            s1=best["s1"], s2=best["s2"], score=best["score"],
        )

        if sp:
            chosen = _select_align_method(
                    self.alignment, self.align_method,
                    rot_deg_thresh=self._auto_rot_deg_thresh,
                    scale_dev_thresh=self._auto_scale_dev_thresh,
                )
            n_cand = len(candidates)
            step_line(
                f"Alignment  (robust {n_cand} cands  "
                f"rot={np.degrees(best['rot']):.3f}°  "
                f"score={best['score']:.3f}  →{chosen})",
                None,
            )
        if self.verbose:
            print(
                f"  Best candidate: frames [{best['frame_start']}–"
                f"{best['frame_end']}]  "
                f"composite={best['composite']:.4f}  "
                f"Alignment: {self.alignment}"
            )
        return self

    def get_shared_alignment(self) -> "SharedAlignment":
        """Extract a SharedAlignment from this pipeline for reuse across a batch.

        Can only be called after ``compute_alignment()`` has completed.
        The returned object holds channel bounds, scrub masks, and alignment
        transform.  Intensity normalisation limits are intentionally *not*
        included — each movie in the batch recomputes them from its own data.
        """
        if self.alignment is None:
            raise RuntimeError(
                "Call compute_alignment() before get_shared_alignment()."
            )
        return SharedAlignment(
            bounds1=self.bounds1.copy(),
            scrub1=self.scrub1.copy(),
            bounds2=self.bounds2.copy() if self.bounds2 is not None else None,
            scrub2=self.scrub2.copy() if self.scrub2 is not None else None,
            alignment=AlignmentParams(
                t1=self.alignment.t1,
                t2=self.alignment.t2,
                rot=self.alignment.rot,
                s1=self.alignment.s1,
                s2=self.alignment.s2,
                score=self.alignment.score,
            ),
            reference_file=str(self.filename) if self.filename is not None else None,
        )

    # ------------------------------------------------------------------ #
    #  Stage 5: align all frames                                           #
    # ------------------------------------------------------------------ #

    def align_frames(
        self,
        frame_min: int = 0,
        frame_max: Optional[int] = None,
        blue_led_frames: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply the alignment transform to all frames and assemble an RGB movie.

        Equivalent to MATLAB ``alignFrames`` → ``alignFramesWithRed`` or
        ``alignFramesNoRed``.

        Parameters
        ----------
        frame_min, frame_max : int, optional
            Frame range to process.  Defaults to all loaded frames.
        blue_led_frames : array-like of bool, optional
            Boolean or 0/1 array of length ``N`` indicating which frames had
            the blue LED on.  Those frames get a blue channel value of 0.5.
            Default: no blue channel (all zeros).

        Returns
        -------
        np.ndarray, shape (H_crop, W_crop, 3, N)
            Normalised RGB movie (float64, values in [0, 1]).
            Channel order: red=0, green=1, blue=2.
        """
        if self.alignment is None:
            self.compute_alignment()

        self._ensure_loaded()
        mov = self._movie
        H, W, total_frames = mov.shape

        if frame_max is None:
            frame_max = total_frames - 1
        frame_max = min(frame_max, total_frames - 1)

        if blue_led_frames is None:
            blue_led_frames = np.zeros(total_frames, dtype=bool)
        blue_led_frames = np.asarray(blue_led_frames, dtype=bool)

        if self.bounds2 is None:
            return self._align_frames_no_red(
                mov[:, :, frame_min: frame_max + 1],
                blue_led_frames[frame_min: frame_max + 1],
            )
        else:
            return self._align_frames_with_red(
                mov[:, :, frame_min: frame_max + 1],
                blue_led_frames[frame_min: frame_max + 1],
            )

    def _align_frames_with_red(
        self,
        imagedata: np.ndarray,
        blue_led_frames: np.ndarray,
    ) -> np.ndarray:
        """Implement MATLAB ``alignFramesWithRed``."""
        if self.verbose:
            a = self.alignment
            print(
                f"--Aligning frames (T: {a.t1:.3f}, {a.t2:.3f}; "
                f"rot: {a.rot:.5f}; S: {a.s1:.5f}, {a.s2:.5f}) ..."
            )

        num_frames = imagedata.shape[2]
        a = self.alignment
        sp = self.show_progress      # shorthand

        # --- Green channel: background subtraction ---
        green = norm_image(
            crop_channel(imagedata, self.bounds1, self.scrub1),
            self._green_min, self._green_max,
        )
        bar_g = ProgressBar(num_frames, label="Bg subtraction  (green)") if sp else None
        t0 = time.perf_counter()
        green = subtract_background(
            green, radius=self.bg_radius, method=self.bg_method,
            n_workers=self.n_workers, backend=self.backend, progress=bar_g,
        )
        if bar_g:
            bar_g.done(elapsed=time.perf_counter() - t0)
        elif self.verbose:
            print("---Green frames ready.")

        # --- Red channel: background subtraction (pre-pad) ---
        red = norm_image(
            crop_channel(imagedata, self.bounds2, self.scrub2),
            self._red_min, self._red_max,
        )
        bar_r = ProgressBar(num_frames, label="Bg subtraction  (red)") if sp else None
        t0 = time.perf_counter()
        # Subtract BEFORE zero-padding (smaller frame → ~5× faster)
        red = subtract_background(
            red, radius=self.bg_radius, method=self.bg_method,
            n_workers=self.n_workers, backend=self.backend, progress=bar_r,
        )
        if bar_r:
            bar_r.done(elapsed=time.perf_counter() - t0)
        elif self.verbose:
            print("---Red background subtracted (pre-pad).")

        # Zero-pad to common power-of-2 size (fast — no bar needed)
        red, green = zero_pad_images(red, green)
        if sp:
            step_line("Zero-padding  (power-of-2)")

        # --- Transform red channel (method chosen by auto-selector) ---
        method = _select_align_method(
            a, self.align_method,
            rot_deg_thresh=self._auto_rot_deg_thresh,
            scale_dev_thresh=self._auto_scale_dev_thresh,
        )

        _METHOD_LABELS = {
            "affine":        "Affine transform  (red)",
            "fft_shift":     "FFT shift  (red)",
            "quick_and_dirty": "Integer shift  (red)",
        }
        label_t = _METHOD_LABELS.get(method, "Transform  (red)")

        if method == "quick_and_dirty":
            # No per-frame progress (essentially instantaneous)
            t0 = time.perf_counter()
            red = integer_shift_image(red, a.t2, a.t1)
            if sp:
                step_line(label_t, time.perf_counter() - t0)

        elif method == "fft_shift":
            # Vectorised — no per-frame bar (single 3D FFT call)
            t0 = time.perf_counter()
            red = fft_shift_image(red, a.t2, a.t1)
            if sp:
                step_line(label_t, time.perf_counter() - t0)

        else:
            # Full affine (default)
            bar_t = ProgressBar(num_frames, label=label_t) if sp else None
            t0 = time.perf_counter()
            red = transform_image(
                red, a.rot, a.s1, a.s2, a.t1, a.t2,
                n_workers=self.n_workers, backend=self.backend, progress=bar_t,
            )
            if bar_t:
                bar_t.done(elapsed=time.perf_counter() - t0)
            elif self.verbose:
                print(", transformed, and ready.")

        if self.verbose:
            print(f"  Align method used: {method}")

        # --- Determine crop range ---
        if self._crop_range is None:
            mean_red   = red.mean(axis=2)
            mean_green = green.mean(axis=2)
            cropping   = (mean_red > 0.01) & (mean_green > 0.01)
            croprows   = cropping.any(axis=1)
            cropcols   = cropping.any(axis=0)

            rowbeg = int(np.argmax(croprows))
            rowend = int(len(croprows) - 1 - np.argmax(croprows[::-1]))
            colbeg = int(np.argmax(cropcols))
            colend = int(len(cropcols) - 1 - np.argmax(cropcols[::-1]))
            self._crop_range = (rowbeg, rowend, colbeg, colend)
        else:
            rowbeg, rowend, colbeg, colend = self._crop_range

        # Assemble RGB movie (H_crop, W_crop, 3, N)
        red_crop   = red  [rowbeg:rowend+1, colbeg:colend+1, :]
        green_crop = green[rowbeg:rowend+1, colbeg:colend+1, :]
        Hc, Wc, _  = red_crop.shape

        if sp:
            step_line(f"Assembling output  ({Hc}×{Wc} px)")
        elif self.verbose:
            print("---Assembling merge ...")

        rgb = np.zeros((Hc, Wc, 3, num_frames), dtype=np.float64)
        rgb[:, :, 0, :] = red_crop    # red
        rgb[:, :, 1, :] = green_crop  # green

        # Blue LED channel
        for i in range(num_frames):
            if i < len(blue_led_frames) and blue_led_frames[i]:
                rgb[:, :, 2, i] = 0.5

        self._rgb_movie = rgb
        return rgb

    def _align_frames_no_red(
        self,
        imagedata: np.ndarray,
        blue_led_frames: np.ndarray,
    ) -> np.ndarray:
        """Implement MATLAB ``alignFramesNoRed``."""
        num_frames = imagedata.shape[2]
        sp = self.show_progress

        green = norm_image(
            crop_channel(imagedata, self.bounds1, self.scrub1),
            self._green_min, self._green_max,
        )
        bar_g = ProgressBar(num_frames, label="Bg subtraction  (green)") if sp else None
        t0 = time.perf_counter()
        green = subtract_background(
            green, radius=self.bg_radius, method=self.bg_method,
            n_workers=self.n_workers, backend=self.backend, progress=bar_g,
        )
        if bar_g:
            bar_g.done(elapsed=time.perf_counter() - t0)
        elif self.verbose:
            print("---Green frames ready.")

        mean_green = green.mean(axis=2)
        cropping   = mean_green > 0.01
        croprows   = cropping.any(axis=1)
        cropcols   = cropping.any(axis=0)

        rowbeg = int(np.argmax(croprows))
        rowend = int(len(croprows) - 1 - np.argmax(croprows[::-1]))
        colbeg = int(np.argmax(cropcols))
        colend = int(len(cropcols) - 1 - np.argmax(cropcols[::-1]))

        green_crop = green[rowbeg:rowend+1, colbeg:colend+1, :]
        Hc, Wc, _  = green_crop.shape

        if self.verbose:
            print("---Assembling merge ...")

        rgb = np.zeros((Hc, Wc, 3, num_frames), dtype=np.float64)
        rgb[:, :, 1, :] = green_crop  # green

        for i in range(num_frames):
            if i < len(blue_led_frames) and blue_led_frames[i]:
                rgb[:, :, 2, i] = 0.5

        self._rgb_movie = rgb
        return rgb

    # ------------------------------------------------------------------ #
    #  Save                                                                #
    # ------------------------------------------------------------------ #

    def save(self, output_path: str | Path) -> None:
        """Save the aligned RGB movie to a TIFF file.

        Parameters
        ----------
        output_path : str or Path
        """
        if self._rgb_movie is None:
            raise RuntimeError("Call align_frames() before save().")
        output_path = Path(output_path)
        if self.verbose:
            print(f"Saving to {output_path} ...")
        save_rgb_tiff(output_path, self._rgb_movie)
        if self.verbose:
            print("Done.")

    # ------------------------------------------------------------------ #
    #  Convenience: run everything                                         #
    # ------------------------------------------------------------------ #

    def run(
        self,
        output_path: Optional[str | Path] = None,
        blue_led_frames: Optional[np.ndarray] = None,
        init_rot: float = 0.0,
        init_s1: float = 1.0,
        init_s2: float = 1.0,
    ) -> np.ndarray:
        """Load, detect channels, align, and optionally save.

        This is the single-call equivalent of running all pipeline stages in
        sequence.

        Parameters
        ----------
        output_path : str or Path, optional
            If provided, the result is also saved to this TIFF path.
        blue_led_frames : array-like, optional
        init_rot, init_s1, init_s2 : float
            Initial alignment parameter guesses.

        Returns
        -------
        np.ndarray, shape (H, W, 3, N)
            Aligned RGB movie.
        """
        self.load()
        self.project()
        self.find_channels()
        self.compute_alignment(init_rot=init_rot, init_s1=init_s1, init_s2=init_s2)
        rgb = self.align_frames(blue_led_frames=blue_led_frames)
        if output_path is not None:
            self.save(output_path)
        return rgb


# ---------------------------------------------------------------------------
# Functional convenience wrappers
# ---------------------------------------------------------------------------

def align_frames(
    filename: str | Path,
    frame_min: int,
    frame_max: int,
    bounds1: np.ndarray,
    scrub1: np.ndarray,
    bounds2: Optional[np.ndarray],
    scrub2: Optional[np.ndarray],
    t1: float,
    t2: float,
    rot: float,
    s1: float,
    s2: float,
    green_min: float,
    green_max: float,
    red_min: float,
    red_max: float,
    blue_led_frames: Optional[np.ndarray] = None,
    crop_range: Optional[Tuple[int, int, int, int]] = None,
    bg_radius: int = 10,
    bg_method: str = "morph_open",
    n_workers: Optional[int] = None,
    backend: str = "auto",
    align_method: str = "auto",
    show_progress: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Functional API: align a frame range with pre-computed parameters.

    Equivalent to MATLAB ``alignFrames(...)``.

    This function is useful when the alignment parameters have been computed
    externally (e.g. from a configuration file) and you just want to apply
    them to a set of frames.

    Parameters
    ----------
    filename : str or Path
    frame_min, frame_max : int   (0-indexed, inclusive)
    bounds1, scrub1 : channel 1 bounds / scrub mask
    bounds2, scrub2 : channel 2 bounds / scrub mask (or None for 1-channel)
    t1, t2 : float   – translation offsets
    rot, s1, s2 : float – rotation / scale
    green_min, green_max : float – intensity limits for green normalisation
    red_min, red_max : float     – intensity limits for red normalisation
    blue_led_frames : array-like, optional
    crop_range : (rowbeg, rowend, colbeg, colend) or None
    bg_radius : int

    Returns
    -------
    rgb_movie : np.ndarray, shape (H_crop, W_crop, 3, N)
    crop_range : (rowbeg, rowend, colbeg, colend)
    """
    imagedata = load_frames(filename, frame_min, frame_max)
    num_frames = imagedata.shape[2]

    if blue_led_frames is None:
        blue_led_frames = np.zeros(num_frames, dtype=bool)
    blue_led_frames = np.asarray(blue_led_frames, dtype=bool)

    a = AlignmentParams(t1=t1, t2=t2, rot=rot, s1=s1, s2=s2)
    pipe = AlignmentPipeline.__new__(AlignmentPipeline)
    pipe._movie = imagedata
    pipe.bounds1 = bounds1
    pipe.scrub1  = scrub1
    pipe.bounds2 = bounds2
    pipe.scrub2  = scrub2
    pipe.alignment = a
    pipe._green_min = green_min
    pipe._green_max = green_max
    pipe._red_min   = red_min
    pipe._red_max   = red_max
    pipe._crop_range   = crop_range
    pipe._rgb_movie    = None
    pipe.bg_radius     = bg_radius
    pipe.bg_method     = bg_method
    pipe.n_workers     = n_workers
    pipe.backend       = backend
    pipe.align_method  = align_method
    pipe.show_progress = show_progress
    pipe.verbose       = False

    if bounds2 is None:
        rgb = pipe._align_frames_no_red(imagedata, blue_led_frames)
    else:
        rgb = pipe._align_frames_with_red(imagedata, blue_led_frames)

    return rgb, pipe._crop_range
