"""
optomerge.segmentation
======================
Locate imaging-channel boundaries in a projected image.

Python equivalent of MATLAB ``findChannelBounds.m`` and its child functions:
``genBoundarySegmentationImage``, ``getChannelBounds``, ``findLine``,
``searchLine``, ``genScrubImage``, ``genScrubLine``, ``boundsWithinImage``,
``convertManualChannelBounds``.

Coordinate convention
---------------------
All bounds returned here are **0-indexed and inclusive**, stored as::

    bounds = np.array([[row_start, row_end],
                       [col_start, col_end]])

This matches the 2×2 format used internally by the MATLAB code (after
converting from 1-indexed).  The :func:`crop_channel` function in
``optomerge.processing`` accepts this format directly.

The *scrub* mask is a boolean ``(H, W)`` array where ``True`` means the pixel
is *outside* the channel and should be zeroed.
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ChannelOrder = Literal[
    "auto",
    "top_green_fils_bottom_red_heads",
    "top_red_heads_bottom_green_fils",
]


def find_channel_bounds(
    max_projection: np.ndarray,
    mean_projection: Optional[np.ndarray] = None,
    channel_order: ChannelOrder = "auto",
    channel_bounds: Optional[np.ndarray] = None,
    segmentation_method: str = "row_profile",
    num_clusters: int = 10,
    cluster_level: int = 4,
    pixel_ratio: float = 3.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Find the pixel boundaries of one or two fluorescence imaging channels.

    Equivalent to MATLAB ``findChannelBounds(primary, ...)``.

    The algorithm:

    1. Segment the mean projection using k-means on log-intensities to obtain
       a binary image of likely channel pixels vs. background.
    2. Determine whether there is one or two imaging channels (or accept a
       manual ``channel_order`` setting).
    3. For each channel, use a multi-scale slanted-line search to find the
       four boundaries (top, bottom, left, right).
    4. Generate a *scrub* mask that zeros out everything outside each channel.

    Parameters
    ----------
    max_projection : np.ndarray, shape (H, W)
        Maximum-intensity projection used for channel-order detection.
    mean_projection : np.ndarray, shape (H, W), optional
        Mean projection used for segmentation.  Falls back to *max_projection*
        if not supplied.
    channel_order : str, optional
        ``'auto'`` (default) – detect from intensity asymmetry.
        ``'top_green_fils_bottom_red_heads'`` – green on top, red on bottom.
        ``'top_red_heads_bottom_green_fils'`` – red on top, green on bottom.
    channel_bounds : np.ndarray, shape (1,4) or (2,4), optional
        Manually supplied bounds ``[x_left, y_top, x_right, y_bottom]`` per
        channel (one row = one channel).  When provided the line-search is
        skipped.
    segmentation_method : str, optional
        ``'quick_kmeans'`` (default) or ``'otsu'``.
    num_clusters : int, optional
        Number of k-means clusters.  Default 10.
    cluster_level : int, optional
        1-indexed cluster level used as channel threshold.  Default 4.
    pixel_ratio : float, optional
        Asymmetry ratio to declare one vs. two channels.  Default 3.
    verbose : bool, optional
        Print progress messages.

    Returns
    -------
    bounds1 : np.ndarray, shape (2, 2)
        ``[[row_start, row_end], [col_start, col_end]]`` for channel 1
        (green/bright channel when two channels are detected).
    scrub1 : np.ndarray, shape (H, W), bool
    bounds2 : np.ndarray or None
        Channel 2 bounds (red channel), or ``None`` if only one channel.
    scrub2 : np.ndarray or None
        Scrub mask for channel 2, or ``None`` if only one channel.
    """
    primary = np.asarray(max_projection, dtype=np.float64)
    if mean_projection is None:
        mean_image = primary
    else:
        mean_image = np.asarray(mean_projection, dtype=np.float64)

    H, W = primary.shape

    # ---- Manual bounds: convert directly, no segmentation needed ----
    if channel_bounds is not None:
        cb = np.atleast_2d(np.asarray(channel_bounds))
        if cb.shape[0] == 1:
            b1, s1 = _convert_manual_bounds(cb[0], W, H)
            return b1, s1, None, None
        b_top, s_top = _convert_manual_bounds(cb[0], W, H)
        b_bot, s_bot = _convert_manual_bounds(cb[1], W, H)
        if channel_order == "top_red_heads_bottom_green_fils":
            is_green_top = False
        elif channel_order == "top_green_fils_bottom_red_heads":
            is_green_top = True
        else:  # auto: brighter half is green
            is_green_top = mean_image[: H // 2, :].mean() >= mean_image[H // 2 :, :].mean()
        return (b_top, s_top, b_bot, s_bot) if is_green_top else (b_bot, s_bot, b_top, s_top)

    # ---- Row-profile method (default): robust axis-aligned rectangular bounds ----
    if segmentation_method in ("row_profile", "profile"):
        return _find_bounds_row_profile(primary, mean_image, channel_order,
                                        pixel_ratio=pixel_ratio, verbose=verbose)

    # ---- Legacy line-search path (k-means/otsu segmentation image) ----
    seg_img_method = "otsu" if segmentation_method == "otsu" else "quick_kmeans"
    channel_pixels_im, boundary_pixels_im = _gen_boundary_segmentation_image(
        mean_image, seg_img_method, num_clusters, cluster_level
    )

    # ---- Determine number of channels and order ----
    num_channels = 2
    is_green_top = True  # green = brighter = first channel in output

    if channel_order != "auto":
        if channel_order == "top_green_fils_bottom_red_heads":
            is_green_top = True
        elif channel_order == "top_red_heads_bottom_green_fils":
            is_green_top = False
        else:
            warnings.warn(
                f"Unknown channel_order '{channel_order}'; falling back to 'auto'."
            )
            channel_order = "auto"

    if channel_order == "auto":
        top_count = channel_pixels_im[: H // 2, :].sum() + 1
        bot_count = channel_pixels_im[H // 2 :, :].sum() + 1
        ratio = top_count / bot_count

        if (1.0 / pixel_ratio) < ratio < pixel_ratio:
            # Roughly equal → one channel
            num_channels = 1
            if verbose:
                print("---Only one channel detected.")
        else:
            # Two channels: the brighter half contains the green channel
            mean_top = primary[: H // 2, :].mean()
            mean_bot = primary[H // 2 :, :].mean()
            is_green_top = mean_top >= mean_bot
            if verbose:
                if is_green_top:
                    print("--Top channel detected to be green.")
                else:
                    print("--Bottom channel detected to be green.")

    bounds_im = boundary_pixels_im

    # ---- Find bounds (line-search) ----
    if num_channels == 1:
        bounds1, scrub1, _, _ = _get_channel_bounds(
            bounds_im, 0, H - 1, 0, W - 1, False, np.array([0, 0])
        )
        return bounds1, scrub1, None, None

    else:  # two channels
        if verbose:
            print("--Searching for channel boundaries...")

        b_top, s_top, close_chans, old_bot = _get_channel_bounds(
            bounds_im, 0, H // 2 - 1, 0, W - 1, False, np.array([0, 0])
        )
        if not close_chans:
            b_bot, s_bot, _, _ = _get_channel_bounds(
                bounds_im, H // 2, H - 1, 0, W - 1, False, old_bot
            )
        else:
            # Fallback: split image evenly
            b_bot = np.array([[H // 2, H - 1], [0, W - 1]])
            s_bot = _gen_scrub_image(
                np.array([H // 2, H // 2]),
                np.array([H - 1, H - 1]),
                np.array([0, 0]),
                np.array([W - 1, W - 1]),
                W, H,
            )

        # Assign green = brighter, red = dimmer
        if is_green_top:
            bounds1, scrub1 = b_top, s_top
            bounds2, scrub2 = b_bot, s_bot
        else:
            bounds1, scrub1 = b_bot, s_bot
            bounds2, scrub2 = b_top, s_top

        return bounds1, scrub1, bounds2, scrub2


# ---------------------------------------------------------------------------
# Row-profile segmentation (default)
# ---------------------------------------------------------------------------
#
# For OptoSplit-style hardware the channels are two horizontal bands separated by
# a dark gap. Rather than a fragile slanted-line search on a noisy k-means mask,
# we read the geometry directly off the *intensity profiles*: a per-row profile
# locates the gap between the two bands and each band's tight top/bottom extent;
# a per-column profile trims the left/right extent. The result is an axis-aligned
# rectangle per channel — scale-invariant (fractional thresholds, no magic
# constants) and robust to uneven brightness/saturation that defeats k-means.


def _smooth1d(x: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smooth (odd window; edges handled by reflection)."""
    if window <= 1:
        return x.astype(np.float64)
    window = int(window) | 1  # force odd
    pad = window // 2
    xp = np.pad(x.astype(np.float64), pad, mode="reflect")
    kernel = np.ones(window) / window
    return np.convolve(xp, kernel, mode="valid")


def _band_extent(profile: np.ndarray, lo: int, hi: int, trim_frac: float) -> Tuple[int, int]:
    """Tight [start, end] (inclusive) within ``[lo, hi]`` where ``profile`` rises
    above ``trim_frac`` of its local dynamic range (trims dark margins)."""
    lo = max(0, int(lo))
    hi = min(len(profile) - 1, int(hi))
    seg = profile[lo:hi + 1]
    if seg.size == 0:
        return lo, hi
    pmin, pmax = float(seg.min()), float(seg.max())
    if pmax - pmin < 1e-12:
        return lo, hi
    thr = pmin + trim_frac * (pmax - pmin)
    above = np.where(seg >= thr)[0]
    if above.size == 0:
        return lo, hi
    return lo + int(above[0]), lo + int(above[-1])


def _rect_scrub(r0: int, r1: int, c0: int, c1: int, W: int, H: int) -> np.ndarray:
    """Boolean mask, ``True`` outside the rectangle ``[r0:r1, c0:c1]`` (inclusive)."""
    scrub = np.ones((H, W), dtype=bool)
    scrub[r0:r1 + 1, c0:c1 + 1] = False
    return scrub


def _find_split(row_prof: np.ndarray, H: int, search_frac: float = 0.4) -> Tuple[int, np.ndarray]:
    """Row of the gap between the two channels: the deepest valley of the smoothed
    row profile within the central ``search_frac`` of the image."""
    smoothed = _smooth1d(row_prof, max(3, H // 50))
    lo = max(1, int(H * (0.5 - search_frac / 2)))
    hi = min(H - 2, int(H * (0.5 + search_frac / 2)))
    if hi <= lo:
        return H // 2, smoothed
    split = lo + int(np.argmin(smoothed[lo:hi + 1]))
    return split, smoothed


def _find_bounds_row_profile(
    max_proj: np.ndarray,
    mean_proj: np.ndarray,
    channel_order: str,
    trim_frac: float = 0.2,
    gap_search_frac: float = 0.4,
    extent_clip_pct: float = 99.5,
    pixel_ratio: float = 3.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Detect one or two channels from row/column intensity profiles.

    The gap between the two channels is found on the smooth *mean* projection.
    The per-channel *extent* (tight top/bottom/left/right bounds) is measured on
    the *max* projection instead -- a row/column is "lit" if it has signal in
    any frame -- after clipping the brightest ``extent_clip_pct`` percentile so a
    cosmic ray / hot pixel (which the max projection accumulates over all frames)
    cannot make a dark row look occupied.

    Returns ``(bounds1, scrub1, bounds2, scrub2)`` in the same convention as
    :func:`find_channel_bounds` (channel 1 = green/bright).
    """
    H, W = mean_proj.shape
    row_prof = mean_proj.mean(axis=1)
    split, smoothed = _find_split(row_prof, H, gap_search_frac)

    # Hot-pixel-robust max projection used for measuring channel extent.
    clip_val = np.percentile(max_proj, extent_clip_pct)
    robust_max = np.minimum(np.asarray(max_proj, dtype=np.float64), clip_val)
    row_ext = robust_max.mean(axis=1)

    # -- one vs two channels --
    if channel_order in ("top_green_fils_bottom_red_heads",
                         "top_red_heads_bottom_green_fils"):
        num_channels = 2
    else:  # auto: a real gap has a deep valley relative to both bands
        top_peak = float(smoothed[:split].max()) if split > 0 else 0.0
        bot_peak = float(smoothed[split:].max()) if split < H else 0.0
        valley = float(smoothed[split])
        span = max(float(smoothed.max()) - float(smoothed.min()), 1e-9)
        two = (top_peak - valley) > 0.25 * span and (bot_peak - valley) > 0.25 * span
        num_channels = 2 if two else 1
        if verbose:
            print(f"--Row-profile: {'two' if two else 'one'} channel(s) "
                  f"(split row {split}).")

    def _band(lo, hi):
        # Row/column extent from the hot-pixel-robust max projection.
        r0, r1 = _band_extent(row_ext, lo, hi, trim_frac)
        col_ext = robust_max[r0:r1 + 1, :].mean(axis=0)
        c0, c1 = _band_extent(col_ext, 0, W - 1, trim_frac)
        return np.array([[r0, r1], [c0, c1]]), _rect_scrub(r0, r1, c0, c1, W, H)

    if num_channels == 1:
        b1, s1 = _band(0, H - 1)
        return b1, s1, None, None

    b_top, s_top = _band(0, split)
    b_bot, s_bot = _band(split, H - 1)

    if channel_order == "top_green_fils_bottom_red_heads":
        is_green_top = True
    elif channel_order == "top_red_heads_bottom_green_fils":
        is_green_top = False
    else:  # auto: the brighter half is the green channel
        is_green_top = max_proj[:split, :].mean() >= max_proj[split:, :].mean()

    if is_green_top:
        return b_top, s_top, b_bot, s_bot
    return b_bot, s_bot, b_top, s_top


# ---------------------------------------------------------------------------
# Segmentation image generation
# ---------------------------------------------------------------------------

def _gen_boundary_segmentation_image(
    primary: np.ndarray,
    method: str,
    num_clusters: int,
    cluster_level: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate channel-pixel and boundary-pixel images.

    Equivalent to MATLAB inner function ``genBoundarySegmentationImage``.

    Returns
    -------
    channel_pixels_image : np.ndarray, bool, shape (H, W)
        True where pixel intensity is at or above ``cluster_level``-th cluster.
    boundary_pixels_image : np.ndarray, bool, shape (H, W)
        True where pixel intensity is at or below the average of the lowest
        two cluster centres (i.e. likely background / channel edge).
    """
    if method == "quick_kmeans":
        cluster_im = np.log(primary + 1.0)
        flat = cluster_im.ravel().astype(np.float64)

        # Guard against very flat images
        if flat.std() < 1e-10:
            ones = np.ones(primary.shape, dtype=bool)
            return ones, ~ones

        try:
            _, labels = kmeans2(flat, num_clusters, minit="points", seed=42)
            # Recover cluster centres as means over assigned pixels
            centres = np.array([
                flat[labels == k].mean() if np.any(labels == k) else np.nan
                for k in range(num_clusters)
            ])
            # Remove any empty clusters
            centres = np.sort(centres[~np.isnan(centres)])
        except Exception:
            # Fallback: Otsu threshold
            method = "otsu"

    if method == "otsu":
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(primary)
        channel_pixels_image = primary > thresh
        boundary_pixels_image = primary <= thresh
        return channel_pixels_image.astype(bool), boundary_pixels_image.astype(bool)

    # Build the two binary images from k-means result
    cluster_im = np.log(primary + 1.0)  # re-use if method was quick_kmeans
    threshold_boundary = (centres[0] + centres[1]) / 2.0
    boundary_pixels_image = cluster_im <= threshold_boundary

    cl_idx = min(cluster_level - 1, len(centres) - 1)  # convert to 0-indexed
    threshold_channel = centres[cl_idx]
    channel_pixels_image = cluster_im >= threshold_channel

    return channel_pixels_image.astype(bool), boundary_pixels_image.astype(bool)


# ---------------------------------------------------------------------------
# Channel bounds (single channel in a given row range)
# ---------------------------------------------------------------------------

def _get_channel_bounds(
    im: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    close_channels: bool,
    old_bot: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
    """Find the four boundaries of one channel within the given row/col range.

    Equivalent to MATLAB inner function ``getChannelBounds``.

    Parameters
    ----------
    im : np.ndarray, shape (H, W)
        Binary boundary image (True = background / edge pixel).
    row_min, row_max : int
        Search range for row boundaries (0-indexed, inclusive).
    col_min, col_max : int
        Search range for column boundaries (0-indexed, inclusive).
    close_channels : bool
        If True the channels are assumed to be directly adjacent.
    old_bot : np.ndarray, shape (2,)
        Bottom boundary coordinates from the previous (top) channel search.

    Returns
    -------
    bounds : np.ndarray, shape (2, 2)
        ``[[row_start, row_end], [col_start, col_end]]``
    scrub : np.ndarray, shape (H, W), bool
    close_channels : bool
    bot : np.ndarray, shape (2,)   – bottom boundary coords found
    """
    H, W = im.shape
    col_range = np.arange(col_min, col_max + 1)
    row_range = np.arange(row_min, row_max + 1)

    if not close_channels:
        # Find bottom boundary
        bot, best_bot_score = _find_line(im, row_max, col_range, "bot")
        # Find top boundary
        top, _ = _find_line(im, row_min, col_range, "top")
        if best_bot_score < 50:
            # Bad score → use previous channel's bottom + 1
            top = old_bot + 1

        # Transpose to search left/right (treat column direction as "top/bot")
        im_T = im.T
        left, _ = _find_line(im_T, col_min, row_range, "top")
        right, _ = _find_line(im_T, col_max, row_range, "bot")

        bounds_both = np.array([top, bot, left, right])
        bounds = _bounds_within_image(W, H, bounds_both)
        scrub = _gen_scrub_image(top, bot, left, right, W, H)

    return bounds, scrub, close_channels, bot


# ---------------------------------------------------------------------------
# Line-finding (multi-scale search)
# ---------------------------------------------------------------------------

def _find_line(
    im: np.ndarray,
    start_coord: int,
    line_range: np.ndarray,
    line_opt: str,
) -> Tuple[np.ndarray, float]:
    """Find the best slanted boundary line via a coarse-to-fine search.

    Equivalent to MATLAB inner function ``findLine``.

    Parameters
    ----------
    im : np.ndarray, shape (H, W)
        Binary boundary image.
    start_coord : int
        Initial row coordinate (0-indexed) for the search centre.
    line_range : np.ndarray, shape (N,)
        Column indices (0-indexed) over which to evaluate the line score.
    line_opt : str
        ``'top'`` or ``'bot'``.

    Returns
    -------
    coords : np.ndarray, shape (2,)  – [left_row, right_row] end-points
    best_score : float
    """
    # Multi-scale search: (coord_range, step)
    search_schedule = [
        (100, 5),
        (30,  3),
        (5,   1),
        (2,   0.2),
    ]

    best_left = float(start_coord)
    best_right = float(start_coord)
    best_score = 0.0

    for coord_range, step in search_schedule:
        best_left, best_right, best_score = _search_line(
            im, best_left, best_right, line_range, coord_range, step, line_opt
        )

    return np.array([best_left, best_right]), best_score


def _search_line(
    im: np.ndarray,
    init_left: float,
    init_right: float,
    line_range: np.ndarray,
    coord_range: float,
    coord_step: float,
    line_option: str,
) -> Tuple[float, float, float]:
    """Score slanted lines and return the best one.

    Equivalent to MATLAB inner function ``searchLine``.

    A "line" is defined by two row endpoints (``left_coord``, ``right_coord``)
    at the leftmost and rightmost columns in *line_range*.  For each column
    position the row is linearly interpolated.  The score is the sum of
    intensity differences between the row below and the row above the line.

    Parameters
    ----------
    im : np.ndarray, shape (H, W)
    init_left, init_right : float
        Current best endpoint row coordinates.
    line_range : np.ndarray
        Column indices to evaluate.
    coord_range : float
        Half-width of the search window around the current best.
    coord_step : float
        Step size within the search window.
    line_option : str
        ``'top'`` → negate score (want high-above / low-below for a top edge).
        ``'bot'`` → keep score as-is (want low-above / high-below).

    Returns
    -------
    best_left, best_right : float
    best_score : float
    """
    H, W = im.shape
    best_left = init_left
    best_right = init_right
    best_score = 0.0

    im_f = im.astype(np.float64)

    left_candidates = np.arange(
        init_left - coord_range, init_left + coord_range + 1e-9, coord_step
    )
    right_candidates = np.arange(
        init_right - coord_range, init_right + coord_range + 1e-9, coord_step
    )

    n_cols = len(line_range)
    col_indices = line_range.astype(float)

    for left_coord in left_candidates:
        for right_coord in right_candidates:
            score = 0.0
            # Line row at each column (linear interpolation, 0-indexed)
            if n_cols > 1:
                line_rows = left_coord + (right_coord - left_coord) / (n_cols - 1) * np.arange(n_cols)
            else:
                line_rows = np.array([left_coord])
            line_rows_r = np.round(line_rows).astype(int)

            for ci, row_r in enumerate(line_rows_r):
                if row_r <= 0:
                    score -= 0.2  # at top edge: slight penalty
                elif row_r >= H - 1:
                    score += 0.2  # at bottom edge: slight benefit
                else:
                    col_idx = int(col_indices[ci])
                    # intensity below minus intensity above the line
                    score += im_f[row_r + 1, col_idx] - im_f[row_r - 1, col_idx]

            if line_option == "top":
                score = -score

            if score > best_score:
                best_left = left_coord
                best_right = right_coord
                best_score = score

    return best_left, best_right, best_score


# ---------------------------------------------------------------------------
# Scrub image generation
# ---------------------------------------------------------------------------

def _gen_scrub_image(
    top_bounds: np.ndarray,
    bot_bounds: np.ndarray,
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    im_width: int,
    im_height: int,
) -> np.ndarray:
    """Generate a boolean mask where ``True`` = outside the channel.

    Equivalent to MATLAB inner function ``genScrubImage``.

    The four boundary lines are slanted lines defined by two row (or column)
    endpoints.

    Parameters
    ----------
    top_bounds, bot_bounds : np.ndarray, shape (2,)
        [left_row, right_row] endpoints for the top/bottom boundary.
    left_bounds, right_bounds : np.ndarray, shape (2,)
        [top_col, bot_col] endpoints for the left/right boundary (image is
        transposed for these, so the interpretation is the same).
    im_width, im_height : int

    Returns
    -------
    np.ndarray, shape (H, W), dtype bool
    """
    scrub = np.zeros((im_height, im_width), dtype=bool)

    top_line = _gen_scrub_line(top_bounds, im_width)
    top_line = np.clip(top_line, 0, im_height - 1)

    bot_line = _gen_scrub_line(bot_bounds, im_width)
    bot_line = np.clip(bot_line, 0, im_height - 1)

    left_line = _gen_scrub_line(left_bounds, im_height)
    left_line = np.clip(left_line, 0, im_width - 1)

    right_line = _gen_scrub_line(right_bounds, im_height)
    right_line = np.clip(right_line, 0, im_width - 1)

    rows = np.arange(im_height)
    cols = np.arange(im_width)

    for i in rows:
        for j in cols:
            if (
                i < top_line[j]
                or i > bot_line[j]
                or j < left_line[i]
                or j > right_line[i]
            ):
                scrub[i, j] = True

    return scrub


def _gen_scrub_image_fast(
    top_bounds: np.ndarray,
    bot_bounds: np.ndarray,
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    im_width: int,
    im_height: int,
) -> np.ndarray:
    """Vectorised equivalent of :func:`_gen_scrub_image` (no Python loops)."""
    top_line = _gen_scrub_line(top_bounds, im_width)   # (W,)
    bot_line = _gen_scrub_line(bot_bounds, im_width)
    top_line = np.clip(top_line, 0, im_height - 1)
    bot_line = np.clip(bot_line, 0, im_height - 1)

    left_line = _gen_scrub_line(left_bounds, im_height)  # (H,)
    right_line = _gen_scrub_line(right_bounds, im_height)
    left_line = np.clip(left_line, 0, im_width - 1)
    right_line = np.clip(right_line, 0, im_width - 1)

    rows = np.arange(im_height)[:, np.newaxis]   # (H, 1)
    cols = np.arange(im_width)[np.newaxis, :]    # (1, W)

    scrub = (
        (rows < top_line[np.newaxis, :])   # rows < top_line[col]
        | (rows > bot_line[np.newaxis, :])
        | (cols < left_line[:, np.newaxis])  # cols < left_line[row]
        | (cols > right_line[:, np.newaxis])
    )
    return scrub


def _gen_scrub_line(coords: np.ndarray, length: int) -> np.ndarray:
    """Linearly interpolate a slanted boundary line.

    Equivalent to MATLAB ``genScrubLine``.

    Parameters
    ----------
    coords : np.ndarray, shape (2,)
        ``[start_coord, end_coord]`` – row (or column) positions at the first
        and last position along *length*.
    length : int

    Returns
    -------
    np.ndarray of int, shape (length,)
    """
    c0, c1 = float(coords[0]), float(coords[1])
    t = np.arange(1, length + 1, dtype=np.float64)  # 1-indexed to match MATLAB
    line = np.round((c1 - c0) / (length - 1) * (t - 1) + c0).astype(int)
    return line


# ---------------------------------------------------------------------------
# Bounds utilities
# ---------------------------------------------------------------------------

def _bounds_within_image(
    im_width: int,
    im_height: int,
    bounds_both_sides: np.ndarray,
) -> np.ndarray:
    """Clip found boundary coordinates to image extent.

    Equivalent to MATLAB ``boundsWithinImage``.

    Parameters
    ----------
    im_width, im_height : int
    bounds_both_sides : np.ndarray, shape (4, 2)
        Rows are [top, bot, left, right]; each row has [left_val, right_val].

    Returns
    -------
    np.ndarray, shape (2, 2) : ``[[row_start, row_end], [col_start, col_end]]``
    """
    top = bounds_both_sides[0]
    bot = bounds_both_sides[1]
    left = bounds_both_sides[2]
    right = bounds_both_sides[3]

    row_start = int(np.clip(np.round(top.min()), 0, im_height - 1))
    row_end   = int(np.clip(np.round(bot.max()), 0, im_height - 1))
    col_start = int(np.clip(np.round(left.min()), 0, im_width - 1))
    col_end   = int(np.clip(np.round(right.max()), 0, im_width - 1))

    return np.array([[row_start, row_end], [col_start, col_end]])


def _convert_manual_bounds(
    channel_bounds: np.ndarray,
    im_width: int,
    im_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert user-supplied ``[x_left, y_top, x_right, y_bottom]`` bounds.

    Equivalent to MATLAB ``convertManualChannelBounds``.

    Parameters
    ----------
    channel_bounds : array-like, shape (4,)
        ``[x_left, y_top, x_right, y_bottom]`` in **0-indexed** pixel
        coordinates.

    Returns
    -------
    bounds : np.ndarray, shape (2, 2)
    scrub  : np.ndarray, shape (H, W), bool
    """
    x_left, y_top, x_right, y_bottom = [int(v) for v in channel_bounds]
    # Map to [row_start, row_end] / [col_start, col_end]
    bounds = np.array([[y_top, y_bottom], [x_left, x_right]])

    top_bounds   = np.array([y_top,    y_top])
    bot_bounds   = np.array([y_bottom, y_bottom])
    left_bounds  = np.array([x_left,   x_left])
    right_bounds = np.array([x_right,  x_right])

    scrub = _gen_scrub_image_fast(
        top_bounds, bot_bounds, left_bounds, right_bounds, im_width, im_height
    )
    return bounds, scrub
