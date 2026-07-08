"""
optomerge.feature_registration
================================
FeatureDistanceAligner: a registration strategy tailored to *point-vs-line*
channels -- e.g. molecular "heads" that sit on gliding filaments and co-move
with them.

Why a separate algorithm
------------------------
Phase correlation aligns two images by their shared texture. It is unreliable
when the two channels image *different-looking* structures: a point-like head in
one channel and a line-like filament in the other share almost no texture, so the
correlation peak is weak and easily captured by noise (this is the "seeing
double" failure on head/filament OptoSplit data).

But those two structures are, physically, *one object*: in every frame the head
sits on the filament (at a tip or somewhere along it). That gives a direct,
geometry-based objective -- find the translation that makes the detected heads
land on the filaments -- which does not depend on the two channels looking alike.

How it works
------------
For a set of sampled frames:

1. Detect head centroids in the moving (red) channel by simple thresholding +
   connected components.
2. Build a distance transform of the filament mask in the reference (green)
   channel: at each pixel, the distance to the nearest filament pixel.
3. Score a candidate translation by the mean distance from every head centroid
   (shifted by that translation) to the nearest filament. Minimising that mean
   distance over a bounded grid gives the registration.

Because the head is on the filament in *every* frame, the correct translation
drives that mean distance toward zero regardless of *where* along the filament
the head sits -- so it is robust to tip- vs mid-filament labels.

This is deliberately the simplest thing that works: threshold-based detection,
translation-only, brute-force grid search. Rotation/scale and better detectors
(e.g. borrowing FASTrack) are natural extensions kept out of the first cut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from .registration import Aligner
from .transform import Transform

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .channel import Channel


class FeatureDistanceAligner(Aligner):
    """Register point-like heads onto line-like filaments by distance minimisation.

    Parameters
    ----------
    max_shift : float
        Half-width of the translation search window, in pixels. The grid spans
        ``[-max_shift, +max_shift]`` on each axis. Must be > 0.
    step : float
        Grid spacing in pixels for the coarse search (a sub-pixel refinement pass
        interpolates around the coarse optimum).
    head_sigma : float
        Head threshold, in std-devs above the per-frame mean of the moving
        channel: ``pixel > mean + head_sigma * std``.
    filament_sigma : float
        Filament threshold, in std-devs above the per-frame mean of the reference
        channel.
    min_head_area : int
        Discard head blobs smaller than this many pixels (rejects noise specks).
    max_head_area : int, optional
        Discard head blobs *larger* than this many pixels (rejects filament
        crossings / bleed-through that aren't point-like heads). ``None`` = no
        upper bound.
    filament_min_area : int
        Drop filament-mask connected components smaller than this many pixels
        before building the distance transform. This despeckles the reference
        mask so background noise is not treated as filament -- otherwise every
        stray bright pixel becomes a spurious "nearest filament".
    distance_cap : float
        Robustness threshold, in pixels. A head whose distance to the nearest
        filament exceeds this is treated as an orphan (a red-only object, or a
        filament too faint to detect): its distance is clipped to ``distance_cap``
        in the cost, so it contributes a constant and cannot bias the fit. Also
        defines the inlier set used for the reported score.
    refine : bool
        Run a parabolic sub-pixel refinement around the best grid cell.
    """

    needs_frames = True

    def __init__(
        self,
        max_shift: float = 30.0,
        step: float = 1.0,
        head_sigma: float = 3.0,
        filament_sigma: float = 2.0,
        min_head_area: int = 4,
        max_head_area: Optional[int] = 60,
        filament_min_area: int = 20,
        distance_cap: float = 6.0,
        refine: bool = True,
    ) -> None:
        if max_shift <= 0:
            raise ValueError("FeatureDistanceAligner needs max_shift > 0 to bound the search.")
        self.max_shift = float(max_shift)
        self.step = float(step)
        self.head_sigma = float(head_sigma)
        self.filament_sigma = float(filament_sigma)
        self.min_head_area = int(min_head_area)
        self.max_head_area = None if max_head_area is None else int(max_head_area)
        self.filament_min_area = int(filament_min_area)
        self.distance_cap = float(distance_cap)
        self.refine = bool(refine)

    # -- detection --------------------------------------------------------- #

    def frame_features(self, green: np.ndarray, red: np.ndarray):
        """Detect head centroids + the filament mask in one aligned frame pair.

        Returns ``(head_rows, head_cols, filament_mask)``. ``head_rows`` /
        ``head_cols`` are the centroids of head blobs in ``red`` (empty arrays if
        none pass the area/threshold gate); ``filament_mask`` is the boolean
        filament mask in ``green``. Shared by the alignment objective (:meth:`_detect`)
        and the diagnostic overlay so both use identical thresholds.
        """
        from scipy import ndimage

        g = np.asarray(green, dtype=np.float64)
        r = np.asarray(red, dtype=np.float64)

        # Filament mask, despeckled: keep only components of real size so stray
        # background pixels don't become spurious "nearest filament" targets.
        fil = g > (g.mean() + self.filament_sigma * g.std())
        fil = self._keep_large_components(fil, self.filament_min_area, ndimage)

        empty = np.empty(0, dtype=np.float64)
        hmask = r > (r.mean() + self.head_sigma * r.std())
        if not hmask.any():
            return empty, empty, fil
        labels, n_blobs = ndimage.label(hmask)
        if n_blobs == 0:
            return empty, empty, fil
        areas = ndimage.sum(np.ones_like(hmask), labels, index=range(1, n_blobs + 1))
        keep = [i + 1 for i, a in enumerate(areas)
                if a >= self.min_head_area
                and (self.max_head_area is None or a <= self.max_head_area)]
        if not keep:
            return empty, empty, fil
        cents = np.atleast_2d(np.asarray(ndimage.center_of_mass(hmask, labels, index=keep),
                                         dtype=np.float64))
        return cents[:, 0], cents[:, 1], fil

    @staticmethod
    def _keep_large_components(mask: np.ndarray, min_area: int, ndimage) -> np.ndarray:
        """Return ``mask`` with connected components smaller than ``min_area`` removed."""
        if min_area <= 1 or not mask.any():
            return mask
        labels, n = ndimage.label(mask)
        if n == 0:
            return mask
        areas = np.asarray(ndimage.sum(np.ones_like(mask), labels, index=range(1, n + 1)))
        big = np.flatnonzero(areas >= min_area) + 1  # component labels to keep
        return np.isin(labels, big)

    def _detect(self, reference: "Channel", moving: "Channel"):
        """Per-frame head centroids (moving) + filament distance transforms (reference).

        Returns a list of ``(rows, cols, dt)`` tuples -- one per usable frame --
        where ``rows``/``cols`` are head-centroid coordinates and ``dt`` is the
        distance-to-nearest-filament map for that frame.
        """
        from scipy import ndimage

        green = np.asarray(reference.data, dtype=np.float64)
        red = np.asarray(moving.data, dtype=np.float64)
        if green.ndim == 2:  # a single projection was passed instead of a stack
            green = green[:, :, None]
            red = red[:, :, None]

        per_frame: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for k in range(red.shape[2]):
            rows, cols, fil = self.frame_features(green[:, :, k], red[:, :, k])
            if rows.size == 0 or not fil.any():
                continue
            dt = ndimage.distance_transform_edt(~fil)
            per_frame.append((rows, cols, dt))

        return per_frame

    # -- objective --------------------------------------------------------- #

    def _distances(self, per_frame, t1: float, t2: float) -> np.ndarray:
        """Head-to-nearest-filament distances if the moving channel shifts by (t1, t2).

        ``transform_image`` is an inverse (destination->source) warp: applying it
        with translation ``(t1, t2)`` moves a feature by ``(-t2 row, -t1 col)``
        (verified against the kernel). So a head detected at ``(row, col)`` lands
        at ``(row - t2, col - t1)`` after the transform is applied downstream --
        that is the position whose distance to the filament we must minimise, so
        the fitted ``t`` is the value the pipeline actually applies.
        Out-of-bounds heads are charged ``distance_cap`` so the search cannot
        cheat by pushing heads off the image. Returns one distance per head.
        """
        out = []
        for rows, cols, dt in per_frame:
            rr = np.round(rows - t2).astype(int)
            cc = np.round(cols - t1).astype(int)
            ok = (rr >= 0) & (rr < dt.shape[0]) & (cc >= 0) & (cc < dt.shape[1])
            d = np.full(rows.shape, self.distance_cap, dtype=np.float64)
            if ok.any():
                d[ok] = dt[rr[ok], cc[ok]]
            out.append(d)
        return np.concatenate(out) if out else np.empty(0)

    def _robust_cost(self, per_frame, t1: float, t2: float) -> float:
        """Mean **capped** head-to-filament distance -- robust to orphan heads.

        Each head's distance is clipped to ``distance_cap`` before averaging, so a
        head with no filament nearby (a red-only object, or a filament too faint
        to detect) contributes a constant regardless of the translation and cannot
        bias the fit. The minimum is therefore driven by the heads that *do* sit
        on filaments.
        """
        d = self._distances(per_frame, t1, t2)
        if d.size == 0:
            return float("inf")
        return float(np.minimum(d, self.distance_cap).mean())

    # -- Aligner API ------------------------------------------------------- #

    def align(self, reference: "Channel", moving: "Channel") -> Transform:
        per_frame = self._detect(reference, moving)
        if not per_frame:
            # No detectable features -> no evidence to move; return identity.
            return Transform(score=0.0)

        grid = np.arange(-self.max_shift, self.max_shift + 1e-9, self.step)

        best_cost = float("inf")
        best = (0.0, 0.0)
        costs = np.empty((grid.size, grid.size), dtype=np.float64)
        for i, t2 in enumerate(grid):          # rows
            for j, t1 in enumerate(grid):      # cols
                c = self._robust_cost(per_frame, t1, t2)
                costs[i, j] = c
                if c < best_cost:
                    best_cost = c
                    best = (float(t1), float(t2))

        t1, t2 = best
        if self.refine:
            t1, t2 = self._refine(costs, grid, best)

        # Report quality from the *inliers* -- heads that actually landed on a
        # filament (distance < cap) -- so orphan heads don't drag the score down.
        d = self._distances(per_frame, t1, t2)
        inliers = d[d < self.distance_cap]
        inlier_frac = inliers.size / d.size if d.size else 0.0
        inlier_mean = float(inliers.mean()) if inliers.size else self.distance_cap
        score = inlier_frac / (1.0 + inlier_mean)  # high = many heads, tightly on filament
        return Transform(t1=float(t1), t2=float(t2), rot=0.0, s1=1.0, s2=1.0, score=float(score))

    def _refine(self, costs: np.ndarray, grid: np.ndarray, best) -> Tuple[float, float]:
        """Parabolic sub-pixel refinement of the (t1, t2) optimum on the cost grid."""
        i = int(np.argmin(np.abs(grid - best[1])))  # row axis (t2)
        j = int(np.argmin(np.abs(grid - best[0])))  # col axis (t1)
        t2 = self._parabolic(costs[:, j], grid, i)
        t1 = self._parabolic(costs[i, :], grid, j)
        return t1, t2

    @staticmethod
    def _parabolic(line: np.ndarray, grid: np.ndarray, k: int) -> float:
        """Sub-cell minimum of a 1-D cost slice via 3-point parabola fit."""
        if k <= 0 or k >= line.size - 1:
            return float(grid[k])
        a, b, c = line[k - 1], line[k], line[k + 1]
        denom = a - 2.0 * b + c
        if denom == 0:
            return float(grid[k])
        delta = 0.5 * (a - c) / denom
        delta = float(np.clip(delta, -1.0, 1.0))
        return float(grid[k] + delta * (grid[1] - grid[0]))
