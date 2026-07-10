"""
optomerge.processing
====================
Per-frame image processing primitives.

Python equivalents
------------------
* ``normImage.m``          → :func:`norm_image`
* ``subtractBackground.m`` → :func:`subtract_background`
* ``cropChannel.m``        → :func:`crop_channel`

Backend selection
-----------------
``subtract_background`` supports two computational backends, chosen via the
``backend`` parameter:

``'auto'`` (default)
    Uses OpenCV (cv2) when available, otherwise falls back to scipy/skimage.
    OpenCV operates in float32 internally and is typically 5–20× faster than
    the scipy equivalent for the morph_open method.

``'cv2'``
    Forces the OpenCV path.  Raises ``ImportError`` if cv2 is not installed.

``'scipy'``
    Forces the scipy/skimage path.  The MATLAB-equivalent reference
    implementation.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


# ---------------------------------------------------------------------------
# OpenCV availability probe
# ---------------------------------------------------------------------------

try:
    import cv2 as _cv2          # type: ignore
    _HAS_CV2: bool = True
except ImportError:
    _cv2 = None                 # type: ignore
    _HAS_CV2 = False

#: Emit the "install OpenCV for a big speedup" notice at most once per process.
_WARNED_NO_CV2 = False


def _warn_missing_cv2_once() -> None:
    """One-time performance warning when the slow scipy backend is auto-selected.

    OpenCV is an optional dependency (``pip install optomerge[cv2]``) but makes
    the background-subtraction / transform kernels ~5-20x faster on the morph
    path. We warn once, and only on the auto backend, so users on the slow path
    know there is a free speedup available -- without nagging those who chose
    scipy deliberately (``backend="scipy"``).
    """
    global _WARNED_NO_CV2
    if _WARNED_NO_CV2:
        return
    _WARNED_NO_CV2 = True
    warnings.warn(
        "OpenCV (cv2) not found -- using the slower scipy background-subtraction "
        "backend. Installing it (pip install opencv-python-headless, or "
        "optomerge[cv2]) typically speeds the dominant step up by 5-20x.",
        RuntimeWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def norm_image(
    image: np.ndarray,
    image_min: float,
    image_max: float,
) -> np.ndarray:
    """Linearly rescale *image* to the range ``[0, 1]``.

    Values below *image_min* are clipped to 0; values above *image_max* are
    clipped to 1.

    Equivalent to MATLAB ``normImage(image, imagemin, imagemax)``.

    Parameters
    ----------
    image : np.ndarray
        Any shape; values are treated as floating-point.
    image_min, image_max : float
        Intensity levels that map to 0 and 1 respectively.

    Returns
    -------
    np.ndarray, same shape as *image*, dtype float64.
    """
    image = np.asarray(image, dtype=np.float64)
    if image_max == image_min:
        return np.zeros_like(image)
    normed = (image - image_min) / (image_max - image_min)
    return np.clip(normed, 0.0, 1.0)


def auto_norm_image(image: np.ndarray, exclude_fraction: float = 0.0) -> np.ndarray:
    """Normalise to [0, 1] using the observed min/max of *image*.

    Parameters
    ----------
    image : np.ndarray
    exclude_fraction : float, optional
        If >0, clip this fraction of the histogram at both ends before
        normalising (robust normalisation).

    Returns
    -------
    np.ndarray, dtype float64.
    """
    image = np.asarray(image, dtype=np.float64)
    if exclude_fraction > 0:
        lo = np.percentile(image, 100.0 * exclude_fraction)
        hi = np.percentile(image, 100.0 * (1 - exclude_fraction))
    else:
        lo = image.min()
        hi = image.max()
    return norm_image(image, lo, hi)


# ---------------------------------------------------------------------------
# Background subtraction — internal per-frame workers
# ---------------------------------------------------------------------------

def _make_morph_open_scipy(radius: int):
    """Return a per-frame morph-open function using scipy + skimage."""
    from scipy.ndimage import grey_opening
    from skimage.morphology import disk
    se = disk(radius)

    def _worker(frame: np.ndarray) -> np.ndarray:
        bg = grey_opening(frame, footprint=se)
        return frame - bg

    return _worker


def _make_morph_open_cv2(radius: int):
    """Return a per-frame morph-open function using cv2 (float32).

    cv2.MORPH_ELLIPSE at (2r+1) × (2r+1) closely approximates skimage's
    disk SE.  Operating in float32 introduces a max error of < 6 × 10⁻⁸
    for [0, 1]-normalised data — negligible for fluorescence microscopy.
    """
    se = _cv2.getStructuringElement(
        _cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )

    def _worker(frame: np.ndarray) -> np.ndarray:
        f32 = frame.astype(np.float32)
        bg  = _cv2.morphologyEx(f32, _cv2.MORPH_OPEN, se)
        return (f32 - bg).astype(np.float64)

    return _worker


def _make_gaussian_scipy(radius: int):
    """Return a per-frame Gaussian blur function using scipy."""
    from scipy.ndimage import gaussian_filter

    def _worker(frame: np.ndarray) -> np.ndarray:
        bg = gaussian_filter(frame, sigma=radius)
        return frame - bg

    return _worker


def _make_gaussian_cv2(radius: int):
    """Return a per-frame Gaussian blur function using cv2 (float32).

    The kernel size is ``6σ + 1`` (rounded up to the nearest odd integer),
    matching scipy's effective filter width.  Float32 introduces a max error
    of < 6 × 10⁻⁸ for [0, 1]-normalised data.
    """
    ksize = int(6 * radius + 1) | 1   # ensure odd

    def _worker(frame: np.ndarray) -> np.ndarray:
        f32 = frame.astype(np.float32)
        bg  = _cv2.GaussianBlur(f32, (ksize, ksize),
                                sigmaX=float(radius), sigmaY=float(radius))
        return (f32 - bg).astype(np.float64)

    return _worker


# ---------------------------------------------------------------------------
# Background subtraction — public API
# ---------------------------------------------------------------------------

def subtract_background(
    movie: np.ndarray,
    radius: int = 10,
    method: str = "morph_open",
    n_workers: int | None = None,
    backend: str = "auto",
    progress: object | None = None,
) -> np.ndarray:
    """Estimate and subtract a slowly-varying background from every frame.

    Equivalent to MATLAB ``subtractBackground.m`` (``morph_open`` branch).

    MATLAB used ``imopen`` with ``strel('ball', 10, intrange/5)``.  For 2-D
    single frames this reduces to a morphological opening with a disk
    structuring element of radius *radius*.

    Frame processing is parallelised over CPU threads by default.

    Parameters
    ----------
    movie : np.ndarray
        Shape ``(H, W)`` or ``(H, W, N)``.
    radius : int, optional
        Radius of the structuring element / Gaussian sigma in pixels.
        Default 10 (matches the MATLAB code).
    method : str, optional
        ``'morph_open'`` (default) – morphological opening background estimate.
        Closest match to the MATLAB reference.
        ``'gaussian'``             – Gaussian blur background estimate
        (σ = *radius*).  Typically 3–10× faster than morph_open.
    n_workers : int or None, optional
        Number of threads.  ``None`` (default) → all logical CPUs.
        Pass ``1`` to disable parallelism.
    backend : str, optional
        ``'auto'``  – cv2 if available, else scipy (default).
        ``'cv2'``   – force OpenCV (``ImportError`` if not installed).
        ``'scipy'`` – force scipy/skimage.
    progress : ProgressBar or None, optional
        If provided, ``progress.advance()`` is called after each frame
        completes so the caller can display a live progress bar.

    Returns
    -------
    np.ndarray, same shape as *movie*, dtype float64.
        Background-subtracted movie; values are *not* clipped.

    Notes
    -----
    The cv2 backend operates in float32 internally.  The maximum rounding
    error vs float64 is < 6 × 10⁻⁸ for [0, 1]-normalised images.
    The output is always float64.
    """
    movie = np.asarray(movie, dtype=np.float64)
    single_frame = movie.ndim == 2
    if single_frame:
        movie = movie[:, :, np.newaxis]

    n_frames = movie.shape[2]
    result = np.zeros_like(movie)

    if n_workers is None:
        from ._parallel import get_default_workers
        n_workers = get_default_workers() or os.cpu_count() or 1

    # ---- choose backend ----
    if backend == "auto":
        use_cv2 = _HAS_CV2
        if not use_cv2 and n_frames > 1:
            _warn_missing_cv2_once()
    elif backend == "cv2":
        if not _HAS_CV2:
            raise ImportError(
                "backend='cv2' requested but OpenCV (cv2) is not installed. "
                "Install it with: pip install opencv-python-headless"
            )
        use_cv2 = True
    elif backend == "scipy":
        use_cv2 = False
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'auto', 'cv2', or 'scipy'.")

    # ---- build per-frame worker ----
    if method == "morph_open":
        worker = _make_morph_open_cv2(radius) if use_cv2 else _make_morph_open_scipy(radius)
    elif method == "gaussian":
        worker = _make_gaussian_cv2(radius) if use_cv2 else _make_gaussian_scipy(radius)
    else:
        raise ValueError(
            f"Unknown background subtraction method: '{method}'. "
            "Choose 'morph_open' or 'gaussian'."
        )

    # ---- dispatch ----
    def _run(i: int) -> None:
        result[:, :, i] = worker(movie[:, :, i])

    if n_workers == 1 or n_frames == 1:
        for i in range(n_frames):
            _run(i)
            if progress is not None:
                progress.advance()
    else:
        with ThreadPoolExecutor(max_workers=min(n_workers, n_frames)) as pool:
            if progress is not None:
                # Use submit+as_completed so each completion triggers a tick
                futures = [pool.submit(_run, i) for i in range(n_frames)]
                for fut in as_completed(futures):
                    fut.result()      # re-raise any worker exception
                    progress.advance()
            else:
                list(pool.map(_run, range(n_frames)))

    if single_frame:
        result = result[:, :, 0]
    return result


# ---------------------------------------------------------------------------
# Channel cropping
# ---------------------------------------------------------------------------

def crop_channel(
    image: np.ndarray,
    bounds: np.ndarray,
    scrub: np.ndarray,
) -> np.ndarray:
    """Crop an image (or movie) to *bounds* and zero-out scrubbed pixels.

    Equivalent to MATLAB ``cropChannel(image, bounds, scrub)``.

    The MATLAB ``bounds`` variable is a 2×2 array:
    ``[[top, bottom], [left, right]]`` (1-indexed, inclusive).  This function
    expects the same layout but **0-indexed and inclusive**, matching the
    output of :func:`optomerge.segmentation.find_channel_bounds`.

    *Scrub* pixels (``scrub == 1``) are set to the minimum non-scrubbed
    intensity before cropping, so that they blend with the dark background
    after normalisation.

    Parameters
    ----------
    image : np.ndarray
        Shape ``(H, W)`` or ``(H, W, N)``.
    bounds : array-like, shape (2, 2)
        ``[[row_start, row_end], [col_start, col_end]]`` (0-indexed, inclusive).
    scrub : np.ndarray
        Shape ``(H, W)`` boolean or 0/1 integer mask.  1 = outside channel.

    Returns
    -------
    np.ndarray
        Cropped array; shape ``(crop_H, crop_W)`` or ``(crop_H, crop_W, N)``.
    """
    bounds = np.asarray(bounds, dtype=int)
    scrub  = np.asarray(scrub,  dtype=bool)
    image  = np.asarray(image,  dtype=np.float64)

    row_start, row_end = bounds[0, 0], bounds[0, 1]
    col_start, col_end = bounds[1, 0], bounds[1, 1]

    single_frame = image.ndim == 2
    if single_frame:
        image = image[:, :, np.newaxis]

    first_frame    = image[:, :, 0]
    non_scrub_vals = first_frame[~scrub]
    min_nonscrub   = float(non_scrub_vals.min()) if non_scrub_vals.size > 0 else 0.0

    result          = image.copy()
    result[scrub, :] = min_nonscrub
    cropped         = result[row_start : row_end + 1, col_start : col_end + 1, :]

    if single_frame:
        cropped = cropped[:, :, 0]
    return cropped
