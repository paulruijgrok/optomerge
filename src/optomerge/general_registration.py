"""
optomerge.general_registration
================================
General-purpose registration strategies for the ``Aligner`` seam.

Where :class:`~optomerge.registration.PhaseCorrelationAligner` (FFT texture
matching) and :class:`~optomerge.feature_registration.FeatureDistanceAligner`
(point-vs-line geometry) are each tuned to a particular kind of channel pair,
the three aligners here cover the *general* two-image registration problem, so a
user with non-OptoSplit or otherwise unusual multichannel data has a method that
fits their case:

* :class:`EccAligner` -- OpenCV's Enhanced Correlation Coefficient. A dense,
  gradient-based intensity method that is invariant to per-channel brightness/
  contrast offsets. Best when the two channels image *similar* structure (so
  their intensities are related by roughly an affine map), e.g. the same beads or
  the same cell outline in two colours.

* :class:`MutualInformationAligner` -- maximises the statistical mutual
  information between the two channels. MI only assumes the intensities are
  *statistically dependent*, not linearly related, so it is the tool of choice
  for genuinely *multimodal* channels whose brightnesses need not track each
  other (the standard workhorse of cross-modal medical registration).

* :class:`FeatureMatchAligner` -- detects and matches keypoints (ORB or SIFT),
  then fits a similarity/affine transform with RANSAC. Handles larger shifts and
  rotations than the intensity methods and is robust to partial overlap, but
  needs enough shared, matchable texture to find correspondences.

All three consume the same ``align(reference, moving) -> Transform`` call as
every other :class:`~optomerge.registration.Aligner` and work on the normalised
channel *projections* (``needs_frames = False``), so they drop into the pipeline
without any other change. Each returns a :class:`~optomerge.transform.Transform`
carrying a method-appropriate quality ``score`` (ECC coefficient, mutual
information, or RANSAC inlier fraction) for QC and acceptance.

Convention
----------
Every method estimates the **destination -> source** (reference-grid -> moving-
grid) affine in top-left-origin pixel coordinates and converts it with
:meth:`Transform.from_affine_matrix`, so the fitted transform is exactly what the
pipeline applies to the moving channel. Library calls that fail to converge (no
texture, too few matches) return an identity transform with ``score = 0`` rather
than raising, so one hard channel pair cannot sink a batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .registration import Aligner
from .transform import Transform

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .channel import Channel


def _require_cv2():
    """Import OpenCV or raise a clear, actionable error."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "The ECC / mutual-information / keypoint aligners require OpenCV. "
            "Install it with:  pip install opencv-python-headless"
        ) from exc
    return cv2


def _as_gray_float32(channel: "Channel") -> np.ndarray:
    """Normalised single-channel float32 projection of a channel (for cv2)."""
    img = np.asarray(channel.normalized(), dtype=np.float32)
    if img.ndim == 3:  # a stack slipped through -> mean-project to 2-D
        img = img.mean(axis=2).astype(np.float32)
    return img


def _as_gray_uint8(channel: "Channel") -> np.ndarray:
    """Normalised single-channel uint8 projection (for keypoint detectors)."""
    return (np.clip(_as_gray_float32(channel), 0.0, 1.0) * 255.0).astype(np.uint8)


class EccAligner(Aligner):
    """Register by maximising the Enhanced Correlation Coefficient (OpenCV).

    ECC fits the warp that makes the moving projection line up with the
    reference under a criterion invariant to affine changes in intensity (gain +
    bias), so unequal channel brightness does not bias the fit. It is a *local*
    optimiser: it refines from the identity, so it wants the true shift to be
    modest (tens of pixels) and the two channels to share visible structure.

    Parameters
    ----------
    motion : {"translation", "euclidean", "affine"}
        Degrees of freedom to fit. ``"euclidean"`` (rotation + translation) is a
        good default for two halves of one camera; ``"translation"`` is the most
        constrained; ``"affine"`` adds scale/shear (use only with real evidence,
        it over-fits easily).
    max_iterations : int
        Maximum ECC iterations.
    eps : float
        Convergence threshold on the ECC increment.
    gauss_blur : int
        Size (pixels, odd) of the Gaussian smoothing OpenCV applies to the
        images and gradients before each iteration; larger widens the capture
        range at the cost of precision. ``0`` disables (mapped to 1).
    max_shift : float
        If > 0, reject a fit whose translation exceeds this many pixels on either
        axis (returns identity, ``score = 0``) -- a guard against ECC wandering
        to a spurious optimum when the channels share little structure.
    """

    needs_frames = False

    _MOTIONS = {"translation", "euclidean", "affine"}

    def __init__(
        self,
        motion: str = "euclidean",
        max_iterations: int = 100,
        eps: float = 1e-5,
        gauss_blur: int = 5,
        max_shift: float = 0.0,
    ) -> None:
        if motion not in self._MOTIONS:
            raise ValueError(f"motion must be one of {sorted(self._MOTIONS)}, got {motion!r}.")
        self.motion = motion
        self.max_iterations = int(max_iterations)
        self.eps = float(eps)
        self.gauss_blur = max(1, int(gauss_blur))
        self.max_shift = float(max_shift)

    def align(self, reference: "Channel", moving: "Channel") -> Transform:
        cv2 = _require_cv2()
        ref = _as_gray_float32(reference)
        mov = _as_gray_float32(moving)

        motion_map = {
            "translation": cv2.MOTION_TRANSLATION,
            "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE,
        }
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.max_iterations, self.eps)
        try:
            # template = reference (destination grid), input = moving (source grid);
            # the returned warpMatrix maps reference -> moving in pixel coords.
            cc, warp = cv2.findTransformECC(
                ref, mov, warp, motion_map[self.motion], criteria, None, self.gauss_blur,
            )
        except cv2.error:
            return Transform(score=0.0)  # did not converge (too little shared texture)

        transform = Transform.from_affine_matrix(warp, ref.shape, score=float(cc))
        if self.max_shift > 0 and (abs(transform.t1) > self.max_shift
                                   or abs(transform.t2) > self.max_shift):
            return Transform(score=0.0)  # implausibly large -> reject
        return transform


class MutualInformationAligner(Aligner):
    """Register by maximising mutual information between the two channels.

    Mutual information measures how much knowing one channel's intensity tells
    you about the other's; it is maximal at registration for *any* consistent
    intensity relationship, not just a linear one. That makes it the robust
    choice for multimodal channels (e.g. a stain vs a label with unrelated
    brightness) where correlation-based methods fail.

    The fit is a bounded coarse-to-fine search: a grid over integer translations
    seeds a derivative-free local refinement over the enabled degrees of freedom
    (translation always; rotation/scale opt-in). MI is evaluated on the joint
    intensity histogram over the region where the warped moving channel is
    in-bounds, so out-of-frame padding cannot inflate the score.

    Parameters
    ----------
    bins : int
        Number of histogram bins per axis for the joint distribution.
    max_shift : float
        Half-width of the translation search / bound, in pixels.
    step : float
        Grid spacing (pixels) for the coarse translation search.
    fit_rotation, fit_scale : bool
        Also refine a small bounded rotation / isotropic scale (off by default).
    rotation_max_deg : float
        Bound on the rotation search, in degrees.
    scale_max_pct : float
        Bound on the isotropic scale search, in percent.
    refine : bool
        Run the local (Nelder-Mead) refinement after the coarse grid. When no
        rotation/scale is requested this sub-pixel-refines the translation.
    """

    needs_frames = False

    def __init__(
        self,
        bins: int = 32,
        max_shift: float = 20.0,
        step: float = 2.0,
        fit_rotation: bool = False,
        fit_scale: bool = False,
        rotation_max_deg: float = 5.0,
        scale_max_pct: float = 5.0,
        refine: bool = True,
    ) -> None:
        if max_shift <= 0:
            raise ValueError("MutualInformationAligner needs max_shift > 0 to bound the search.")
        self.bins = int(bins)
        self.max_shift = float(max_shift)
        self.step = float(step)
        self.fit_rotation = bool(fit_rotation)
        self.fit_scale = bool(fit_scale)
        self.rotation_max_deg = float(rotation_max_deg)
        self.scale_max_pct = float(scale_max_pct)
        self.refine = bool(refine)

    # -- MI objective ------------------------------------------------------ #

    def _mi(self, a: np.ndarray, b: np.ndarray) -> float:
        """Mutual information (nats) of two equal-length intensity samples."""
        if a.size < 16:
            return 0.0
        hist, _, _ = np.histogram2d(a, b, bins=self.bins)
        pxy = hist / hist.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        nz = pxy > 0
        outer = px[:, None] * py[None, :]
        return float((pxy[nz] * np.log(pxy[nz] / outer[nz])).sum())

    def _cost_translation(self, ref: np.ndarray, mov: np.ndarray,
                          t1: float, t2: float) -> float:
        """-MI for an integer translation, over the exact valid overlap.

        Mirrors ``transform_image``'s sign convention (applying translation
        ``(t1, t2)`` samples the moving channel at ``(row - t2, col - t1)``), so
        the seed matches the continuous refinement and the pipeline.
        """
        dy, dx = int(round(t2)), int(round(t1))
        H, W = ref.shape
        # Reference[r, c] is compared with moving[r - dy, c - dx]; keep in-bounds.
        r0, r1 = max(0, dy), min(H, H + dy)
        c0, c1 = max(0, dx), min(W, W + dx)
        if r1 - r0 < 4 or c1 - c0 < 4:
            return 0.0
        ref_crop = ref[r0:r1, c0:c1]
        mov_crop = mov[r0 - dy:r1 - dy, c0 - dx:c1 - dx]
        return -self._mi(ref_crop.ravel(), mov_crop.ravel())

    def _cost_affine(self, ref: np.ndarray, mov: np.ndarray, ones: np.ndarray,
                     t1: float, t2: float, rot: float, s: float) -> float:
        """-MI for a continuous (translation, rotation, isotropic scale) warp."""
        tf = Transform(t1=t1, t2=t2, rot=rot, s1=s, s2=s)
        warped = tf.apply(mov)
        valid = tf.apply(ones) > 0.5  # pixels that sampled inside the moving image
        if valid.sum() < 16:
            return 0.0
        return -self._mi(ref[valid].ravel(), warped[valid].ravel())

    def align(self, reference: "Channel", moving: "Channel") -> Transform:
        ref = _as_gray_float32(reference)
        mov = _as_gray_float32(moving)
        if ref.shape != mov.shape:
            # Intensity methods need a common grid; refuse rather than mis-register.
            return Transform(score=0.0)

        grid = np.arange(-self.max_shift, self.max_shift + 1e-9, self.step)
        best_cost, best = float("inf"), (0.0, 0.0)
        for t2 in grid:
            for t1 in grid:
                c = self._cost_translation(ref, mov, float(t1), float(t2))
                if c < best_cost:
                    best_cost, best = c, (float(t1), float(t2))

        t1, t2 = best
        rot, s = 0.0, 1.0
        if self.refine or self.fit_rotation or self.fit_scale:
            from scipy.optimize import minimize

            ones = np.ones_like(mov)
            rot_max = np.deg2rad(self.rotation_max_deg)
            s_lo, s_hi = 1.0 - self.scale_max_pct / 100.0, 1.0 + self.scale_max_pct / 100.0

            x0 = [t1, t2]
            bounds = [(-self.max_shift, self.max_shift)] * 2
            if self.fit_rotation:
                x0.append(0.0)
                bounds.append((-rot_max, rot_max))
            if self.fit_scale:
                x0.append(1.0)
                bounds.append((s_lo, s_hi))

            def _unpack(x):
                r, sc, i = 0.0, 1.0, 2
                if self.fit_rotation:
                    r = x[i]
                    i += 1
                if self.fit_scale:
                    sc = x[i]
                    i += 1
                return x[0], x[1], r, sc

            def _cost(x):
                a1, a2, r, sc = _unpack(x)
                return self._cost_affine(ref, mov, ones, a1, a2, r, sc)

            res = minimize(_cost, np.asarray(x0, dtype=np.float64), method="Nelder-Mead",
                           bounds=bounds, options={"xatol": 1e-2, "fatol": 1e-4, "maxiter": 1500})
            t1, t2, rot, s = _unpack(res.x)
            score = -_cost(res.x)
        else:
            score = -best_cost

        return Transform(t1=float(t1), t2=float(t2), rot=float(rot),
                         s1=float(s), s2=float(s), score=float(score))


class FeatureMatchAligner(Aligner):
    """Register by matching keypoints (ORB/SIFT) and a RANSAC affine fit.

    Detects distinctive keypoints in both projections, matches their descriptors,
    and robustly fits a transform to the surviving correspondences with RANSAC.
    Unlike the intensity methods this is not a local optimiser, so it copes with
    larger shifts and rotations -- but it needs enough shared, corner-like texture
    in *both* channels to find matches, so it suits richly textured images (beads,
    speckle, cell edges) rather than sparse point sources.

    Parameters
    ----------
    detector : {"orb", "sift"}
        Keypoint detector/descriptor. ORB is fast and license-free; SIFT is more
        robust to scale/contrast changes (available in OpenCV >= 4.4).
    max_features : int
        Maximum keypoints to retain per image.
    motion : {"similarity", "affine"}
        ``"similarity"`` fits rotation + uniform scale + translation
        (``estimateAffinePartial2D``); ``"affine"`` adds shear/anisotropy
        (``estimateAffine2D``). Similarity is the safer default.
    ratio : float
        Lowe ratio-test threshold for accepting a match (lower = stricter).
    min_matches : int
        Minimum good matches required to attempt a fit; below this, identity is
        returned.
    ransac_thresh : float
        RANSAC reprojection tolerance, in pixels.
    max_shift : float
        If > 0, reject a fit whose translation exceeds this many pixels on either
        axis (returns identity, ``score = 0``).
    """

    needs_frames = False

    _DETECTORS = {"orb", "sift"}
    _MOTIONS = {"similarity", "affine"}

    def __init__(
        self,
        detector: str = "orb",
        max_features: int = 2000,
        motion: str = "similarity",
        ratio: float = 0.75,
        min_matches: int = 8,
        ransac_thresh: float = 3.0,
        max_shift: float = 0.0,
    ) -> None:
        if detector not in self._DETECTORS:
            raise ValueError(f"detector must be one of {sorted(self._DETECTORS)}, got {detector!r}.")
        if motion not in self._MOTIONS:
            raise ValueError(f"motion must be one of {sorted(self._MOTIONS)}, got {motion!r}.")
        self.detector = detector
        self.max_features = int(max_features)
        self.motion = motion
        self.ratio = float(ratio)
        self.min_matches = int(min_matches)
        self.ransac_thresh = float(ransac_thresh)
        self.max_shift = float(max_shift)

    def _build_detector(self, cv2):
        if self.detector == "orb":
            return cv2.ORB_create(nfeatures=self.max_features), cv2.NORM_HAMMING
        return cv2.SIFT_create(nfeatures=self.max_features), cv2.NORM_L2

    def align(self, reference: "Channel", moving: "Channel") -> Transform:
        cv2 = _require_cv2()
        ref = _as_gray_uint8(reference)
        mov = _as_gray_uint8(moving)

        det, norm = self._build_detector(cv2)
        kp_ref, des_ref = det.detectAndCompute(ref, None)
        kp_mov, des_mov = det.detectAndCompute(mov, None)
        if des_ref is None or des_mov is None or len(kp_ref) < 2 or len(kp_mov) < 2:
            return Transform(score=0.0)

        matcher = cv2.BFMatcher(norm)
        knn = matcher.knnMatch(des_ref, des_mov, k=2)
        good = [m for pair in knn if len(pair) == 2
                for m, n in [pair] if m.distance < self.ratio * n.distance]
        if len(good) < self.min_matches:
            return Transform(score=0.0)

        # Points as (x, y) = (col, row); from_pts on reference, to_pts on moving,
        # so the estimated M maps reference (destination) -> moving (source).
        ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        mov_pts = np.float32([kp_mov[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        if self.motion == "similarity":
            M, inliers = cv2.estimateAffinePartial2D(
                ref_pts, mov_pts, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh)
        else:
            M, inliers = cv2.estimateAffine2D(
                ref_pts, mov_pts, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh)
        if M is None or inliers is None or int(inliers.sum()) < self.min_matches:
            return Transform(score=0.0)

        inlier_frac = float(inliers.sum()) / len(good)
        transform = Transform.from_affine_matrix(M, ref.shape, score=inlier_frac)
        if self.max_shift > 0 and (abs(transform.t1) > self.max_shift
                                   or abs(transform.t2) > self.max_shift):
            return Transform(score=0.0)
        return transform
