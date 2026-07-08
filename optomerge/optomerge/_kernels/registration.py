"""
optomerge.registration
======================
Sub-pixel image registration via phase correlation combined with an iterative
scale- and rotation-correction search.

Python equivalents
------------------
* ``calculateAlignment.m``  → :func:`calculate_alignment`
* ``calcScaleRotation``     → :func:`_calc_scale_rotation` (internal)
* ``calcFFT2Dalign``        → :func:`_calc_fft2d_align` (internal)

Algorithm overview
------------------
1. **Zero-pad** both images to the same power-of-2 size.
2. **Scale / rotation search** (``_calc_scale_rotation``):
   Iteratively narrow a search window over ``rot``, ``s1`` (x-scale), and
   ``s2`` (y-scale).  For each candidate transform, the image is warped with
   :func:`~optomerge.transform.transform_image` and the FFT phase-correlation
   score is used as the quality metric.
3. **Translational alignment** (``_calc_fft2d_align``):
   Normalised cross-correlation in the Fourier domain (phase correlation),
   followed by sub-pixel refinement via 2-D spline interpolation around the
   correlation peak.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .transform import transform_image, zero_pad_images


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def calculate_alignment(
    im1: np.ndarray,
    im2: np.ndarray,
    init_rot: float = 0.0,
    init_s1: float = 1.0,
    init_s2: float = 1.0,
    max_shift: float = 0.0,
    fit_scale_rotation: bool = True,
) -> Tuple[float, float, float, float, float, float]:
    """Calculate the affine alignment between two images.

    Equivalent to MATLAB ``calculateAlignment(im1, im2, initrot, inits1, inits2)``.

    The function first zero-pads both images to a common power-of-2 size, then
    iteratively searches for the rotation and anisotropic scale factors that
    maximise the phase-correlation score, and finally extracts the translational
    offset.

    Parameters
    ----------
    im1 : np.ndarray, shape (H1, W1)
        Reference image (e.g. the green channel projection).
    im2 : np.ndarray, shape (H2, W2)
        Image to align to *im1* (e.g. the red channel projection).
    init_rot : float, optional
        Initial guess for the rotation angle (radians).  Default 0.
    init_s1 : float, optional
        Initial guess for the x-scale factor.  Default 1.
    init_s2 : float, optional
        Initial guess for the y-scale factor.  Default 1.
    max_shift : float, optional
        If > 0, constrain the phase-correlation peak to translations within
        ``±max_shift`` pixels of the origin (in either axis), so a spurious
        far-off correlation peak cannot win. Use when the true inter-channel
        shift is known to be small (e.g. two halves of one camera frame).
        Default 0 = unconstrained (exact MATLAB behaviour).
    fit_scale_rotation : bool, optional
        If True (default), search for the rotation + anisotropic scale that
        maximise the phase-correlation score. If False, hold rotation/scale at
        the ``init_*`` values and fit translation only -- appropriate when the
        inter-channel rotation/scale is a fixed, near-identity optical property
        (e.g. an OptoSplit), where a free search tends to over-fit and invent a
        spurious rotation.

    Returns
    -------
    t1 : float   – translation along rows    (pixels)
    t2 : float   – translation along columns (pixels)
    rot : float  – rotation angle (radians)
    s1 : float   – x-scale factor
    s2 : float   – y-scale factor
    best_score : float  – phase-correlation peak score
    """
    nim1, nim2 = zero_pad_images(
        np.asarray(im1, dtype=np.float64),
        np.asarray(im2, dtype=np.float64),
    )
    if fit_scale_rotation:
        rot, s1, s2, t1, t2, best_score = _calc_scale_rotation(
            nim1, nim2, init_rot, init_s1, init_s2, max_shift=max_shift
        )
    else:
        # Translation only, at the initial rotation/scale.
        warped = transform_image(nim2, init_rot, init_s1, init_s2)
        best_score, offset = _calc_fft2d_align(nim1, warped, max_shift=max_shift)
        t2, t1 = -offset[0], -offset[1]
        rot, s1, s2 = init_rot, init_s1, init_s2
    return t1, t2, rot, s1, s2, best_score


# ---------------------------------------------------------------------------
# Scale / rotation search
# ---------------------------------------------------------------------------

def _calc_scale_rotation(
    im1: np.ndarray,
    im2: np.ndarray,
    init_rot: float,
    init_s1: float,
    init_s2: float,
    verbose: bool = False,
    max_shift: float = 0.0,
) -> Tuple[float, float, float, float, float, float]:
    """Iterative scale and rotation refinement using phase-correlation.

    Equivalent to MATLAB inner function ``calcScaleRotation``.

    Uses a golden-section-like shrinking search window over three parameters
    (``rot``, ``s1``, ``s2``) independently.  The window shrinks when the
    current best is at the centre of the range (indicating convergence) and
    expands the other parameters when the best is at an edge (indicating the
    other parameters may need a wider range).

    Parameters
    ----------
    im1, im2 : np.ndarray, shape (H, W)
        Zero-padded images of equal size.
    init_rot, init_s1, init_s2 : float
        Initial parameter estimates.
    verbose : bool
        Print a progress dot each iteration.

    Returns
    -------
    rot, s1, s2 : float – converged parameters
    t1, t2 : float      – translation offsets (from last FFT alignment)
    best_score : float
    """
    INIT_SCALE_RANGE = 0.01
    MIN_SCALE_RANGE  = 0.001
    INIT_ROT_RANGE   = 0.01
    MIN_ROT_RANGE    = 0.0001

    curr_s1    = init_s1
    curr_s2    = init_s2
    curr_rot   = init_rot
    curr_s1off = INIT_SCALE_RANGE
    curr_s2off = INIT_SCALE_RANGE
    curr_rotoff = INIT_ROT_RANGE

    best_score = 0.0
    best_offset = np.array([0.0, 0.0])

    while (
        curr_s1off > MIN_SCALE_RANGE
        or curr_s2off > MIN_SCALE_RANGE
        or curr_rotoff > MIN_ROT_RANGE
    ):
        # ---- Search over s1 ----
        if curr_s1off > MIN_SCALE_RANGE:
            s1_range = np.arange(
                curr_s1 - curr_s1off, curr_s1 + curr_s1off + 1e-12, curr_s1off
            )
            local_best_score = 0.0
            best_ind = 0
            for idx, sc in enumerate(s1_range):
                tmp = transform_image(im2, curr_rot, sc, curr_s2)
                score, offset = _calc_fft2d_align(im1, tmp, max_shift=max_shift)
                if score > local_best_score:
                    local_best_score = score
                    best_score = score
                    best_offset = offset
                    best_ind = idx
            curr_s1 = s1_range[best_ind]
            if best_ind == len(s1_range) // 2:  # middle → converging
                curr_s1off /= 2.0
            else:
                curr_s2off  *= 1.2
                curr_rotoff *= 1.2

        # ---- Search over s2 ----
        if curr_s2off > MIN_SCALE_RANGE:
            s2_range = np.arange(
                curr_s2 - curr_s2off, curr_s2 + curr_s2off + 1e-12, curr_s2off
            )
            local_best_score = 0.0
            best_ind = 0
            for idx, sc in enumerate(s2_range):
                tmp = transform_image(im2, curr_rot, curr_s1, sc)
                score, offset = _calc_fft2d_align(im1, tmp, max_shift=max_shift)
                if score > local_best_score:
                    local_best_score = score
                    best_score = score
                    best_offset = offset
                    best_ind = idx
            curr_s2 = s2_range[best_ind]
            if best_ind == len(s2_range) // 2:
                curr_s2off /= 2.0
            else:
                curr_s1off  *= 1.2
                curr_rotoff *= 1.2

        # ---- Search over rot ----
        if curr_rotoff > MIN_ROT_RANGE:
            rot_range = np.arange(
                curr_rot - curr_rotoff, curr_rot + curr_rotoff + 1e-12, curr_rotoff
            )
            local_best_score = 0.0
            best_ind = 0
            for idx, sc in enumerate(rot_range):
                tmp = transform_image(im2, sc, curr_s1, curr_s2)
                score, offset = _calc_fft2d_align(im1, tmp, max_shift=max_shift)
                if score > local_best_score:
                    local_best_score = score
                    best_score = score
                    best_offset = offset
                    best_ind = idx
            curr_rot = rot_range[best_ind]
            if best_ind == len(rot_range) // 2:
                curr_rotoff /= 2.0
            else:
                curr_s1off *= 1.2
                curr_s2off *= 1.2

        if verbose:
            print(".", end="", flush=True)

    if verbose:
        print()

    # MATLAB: t2 = -bestoff(1);  t1 = -bestoff(2)
    t2 = -best_offset[0]
    t1 = -best_offset[1]

    return curr_rot, curr_s1, curr_s2, t1, t2, best_score


# ---------------------------------------------------------------------------
# FFT phase-correlation alignment
# ---------------------------------------------------------------------------

def _peak_index(phase_map: np.ndarray, max_shift: float = 0.0) -> Tuple[int, int]:
    """Return the ``(row, col)`` of the phase-correlation peak.

    With ``max_shift <= 0`` this is the global argmax (exact MATLAB behaviour).
    With ``max_shift > 0`` the search is restricted to translations within
    ``±max_shift`` pixels of the origin along each axis, accounting for FFT
    periodicity (a shift of ``-k`` appears at index ``N - k``). This prevents a
    spurious far-off correlation peak from winning when the true inter-channel
    shift is known to be small.
    """
    if not max_shift or max_shift <= 0:
        return np.unravel_index(int(np.argmax(phase_map)), phase_map.shape)

    H, W = phase_map.shape
    rows = np.arange(H)
    cols = np.arange(W)
    # Unwrap indices to signed shifts: [0..N/2] stay, (N/2..N) become negative.
    urow = np.where(rows <= H // 2, rows, rows - H)
    ucol = np.where(cols <= W // 2, cols, cols - W)
    row_ok = np.abs(urow) <= max_shift
    col_ok = np.abs(ucol) <= max_shift
    mask = np.outer(row_ok, col_ok)
    if not mask.any():  # max_shift smaller than one pixel window; fall back
        return np.unravel_index(int(np.argmax(phase_map)), phase_map.shape)
    masked = np.where(mask, phase_map, -np.inf)
    return np.unravel_index(int(np.argmax(masked)), masked.shape)


def _calc_fft2d_align(
    nim1: np.ndarray,
    nim2: np.ndarray,
    n_neighbor: int = 4,
    fine_step: float = 0.1,
    max_shift: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """Sub-pixel translational alignment via normalised phase correlation.

    Equivalent to MATLAB inner function ``calcFFT2Dalign``.

    Computes the normalised cross-power spectrum of two images and finds the
    peak of the inverse FFT (phase correlation map), then refines to sub-pixel
    accuracy using 2-D spline interpolation.

    Parameters
    ----------
    nim1, nim2 : np.ndarray, shape (H, W)
        Zero-padded images of equal size.
    n_neighbor : int, optional
        Half-size of the interpolation neighbourhood around the peak.
        Default 4 (matches MATLAB).
    fine_step : float, optional
        Step size for fine interpolation grid.  Default 0.1.

    Returns
    -------
    score : float
        Peak value of the interpolated phase-correlation surface.
    offset : np.ndarray, shape (2,)
        ``[row_offset, col_offset]`` sub-pixel shift that maximises alignment.
        A value of ``[0, 0]`` means no translation.
    """
    H, W = nim1.shape

    fnim1 = np.fft.fft2(nim1)
    fnim2 = np.fft.fft2(nim2)

    cross = fnim1 * np.conj(fnim2)
    denom = np.abs(cross)
    # Guard against division by zero
    denom[denom == 0] = 1e-10
    fpower = cross / denom

    phase_map = np.fft.ifft2(fpower).real  # (H, W)

    # Coarse peak (optionally constrained to a small-shift window)
    off1, off2 = _peak_index(phase_map, max_shift=max_shift)  # (row, col)

    # Extract neighbourhood (with circular wrap)
    nn = n_neighbor
    row_nb = np.arange(off1 - nn, off1 + nn + 1)
    col_nb = np.arange(off2 - nn, off2 + nn + 1)
    row_nb = row_nb % H
    col_nb = col_nb % W

    small_surf = phase_map[np.ix_(row_nb, col_nb)]  # (2*nn+1, 2*nn+1)

    # Fine interpolation grid
    coarse_coords = np.arange(-nn, nn + 1, dtype=float)  # length 2*nn+1
    fine_coords = np.arange(-nn, nn + fine_step / 2, fine_step)

    # Build spline on coarse grid
    spline = RectBivariateSpline(coarse_coords, coarse_coords, small_surf, kx=3, ky=3)
    fine_surf = spline(fine_coords, fine_coords)

    score = float(fine_surf.max())
    fine_r_idx, fine_c_idx = np.unravel_index(np.argmax(fine_surf), fine_surf.shape)

    # Combine coarse + fine offsets
    refined_off1 = off1 + fine_coords[fine_r_idx]
    refined_off2 = off2 + fine_coords[fine_c_idx]

    # Centre translations: values > half the image size mean negative shift
    if refined_off1 > H / 2:
        refined_off1 -= H
    if refined_off2 > W / 2:
        refined_off2 -= W

    # Subtract 1 because a value of 1 means no translation (MATLAB convention)
    offset = np.array([refined_off1, refined_off2]) - 1.0

    return score, offset
