"""
optomerge.transform
=====================
Transform: a fitted geometric mapping from one channel onto another.

This replaces the loose ``AlignmentParams`` + scattered ``transform_image``
calls. The mapping knows how to apply itself, so callers never re-derive the
math. Making it an object also lets a future projective/non-rigid model be a
subclass that the rest of the package consumes through the same ``apply`` API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._core import transform_image


@dataclass
class Transform:
    """An affine mapping (translation + rotation + anisotropic scale).

    Attributes mirror the original ``AlignmentParams`` so migration is direct.
    The sign/axis conventions are exactly those of the functional
    ``transform_image(im, rot, sx, sy, tx, ty)`` it wraps, where the original
    pipeline called ``transform_image(red, rot, s1, s2, t1, t2)``. Hence
    ``s1 -> sx``, ``s2 -> sy``, ``t1 -> tx``, ``t2 -> ty``.

    Attributes
    ----------
    t1, t2 : float
        Translation passed as ``tx`` / ``ty`` (pixels).
    rot : float
        Rotation (radians).
    s1, s2 : float
        Scale factors passed as ``sx`` / ``sy``.
    score : float
        Quality of the fit (phase-correlation peak), for diagnostics/QC.
    """

    t1: float = 0.0
    t2: float = 0.0
    rot: float = 0.0
    s1: float = 1.0
    s2: float = 1.0
    score: float = 0.0

    @classmethod
    def from_affine_matrix(
        cls, matrix: np.ndarray, image_shape: tuple, score: float = 0.0
    ) -> "Transform":
        """Build a :class:`Transform` from a 2x3 (or 3x3) pixel-coordinate affine.

        This is the exact inverse of the ``_build_cv2_matrix`` conversion used by
        :func:`~optomerge._kernels.transform.transform_image`, so a transform
        estimated by an external library (OpenCV ECC, ``estimateAffinePartial2D``,
        ...) can be expressed in the package's native convention and applied by
        the pipeline unchanged.

        The input ``matrix`` must be the **destination -> source** (inverse /
        backward) affine in **top-left-origin pixel coordinates** -- i.e. it maps
        a pixel of the *reference* (destination) grid to the location in the
        *moving* (source) grid to sample, exactly what OpenCV returns as its
        ``warpMatrix`` (used with ``WARP_INVERSE_MAP``) and what
        ``estimateAffinePartial2D(reference_pts, moving_pts)`` returns::

            source_pixel = A2 @ dest_pixel + offset

        It is decomposed into the centred-coordinate ``(rot, s1, s2, t1, t2)``
        that :class:`Transform` stores, where ``A2 = [[cos*s1, -sin*s2],
        [sin*s1, cos*s2]]`` and the translation acts about the image centre
        ``(W/2 - 0.5, H/2 - 0.5)``.

        Parameters
        ----------
        matrix : np.ndarray
            The ``2x3`` (or ``3x3`` homogeneous) affine described above.
        image_shape : tuple
            ``(H, W[, ...])`` of the images being registered; only ``H`` and
            ``W`` are used, to convert between corner- and centre-origin.
        score : float
            Quality metric to carry through for QC (e.g. ECC coefficient or
            RANSAC inlier fraction).

        Notes
        -----
        Rotation is read from the first column and the two scales from the column
        norms, so a pure rotation + (possibly anisotropic) scale + translation is
        recovered exactly. A general affine with **shear** cannot be represented
        by this rigid-plus-scale model and is approximated (the shear is
        discarded); estimators used here (Euclidean/similarity) do not introduce
        shear.
        """
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape == (3, 3):
            m = m[:2, :]
        if m.shape != (2, 3):
            raise ValueError(f"Expected a 2x3 or 3x3 affine, got shape {m.shape}.")

        a2 = m[:, :2]
        offset = m[:, 2]
        H, W = int(image_shape[0]), int(image_shape[1])
        centre = np.array([W / 2.0 - 0.5, H / 2.0 - 0.5], dtype=np.float64)
        # Invert  offset = t - A2 @ centre + centre  ->  t = offset + A2 @ centre - centre
        t = offset + a2 @ centre - centre

        s1 = float(np.hypot(a2[0, 0], a2[1, 0]))
        s2 = float(np.hypot(a2[0, 1], a2[1, 1]))
        rot = float(np.arctan2(a2[1, 0], a2[0, 0]))
        return cls(t1=float(t[0]), t2=float(t[1]), rot=rot, s1=s1, s2=s2, score=float(score))

    @property
    def is_identity(self) -> bool:
        return (
            self.t1 == 0.0 and self.t2 == 0.0 and self.rot == 0.0
            and self.s1 == 1.0 and self.s2 == 1.0
        )

    def matrix(self) -> np.ndarray:
        """Return the 3x3 homogeneous matrix form (acts on ``[x, y, 1]``).

        Matches the matrix built inside ``transform_image``.
        """
        c, s = np.cos(self.rot), np.sin(self.rot)
        return np.array(
            [
                [c * self.s1, -s * self.s2, self.t1],
                [s * self.s1,  c * self.s2, self.t2],
                [0.0,          0.0,         1.0],
            ],
            dtype=np.float64,
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Warp ``image`` (2-D ``(H, W)`` or stacked ``(H, W, N)``).

        Wraps the functional ``transform_image``; a GPU/lazy backend would
        override only this method.
        """
        if self.is_identity:
            image = np.asarray(image)
            if np.issubdtype(image.dtype, np.floating):
                return image
            return image.astype(np.float64)
        return transform_image(image, self.rot, self.s1, self.s2, self.t1, self.t2)

    def rescaled(self, factor: float) -> "Transform":
        """Return a copy with translations scaled (e.g. scrub-space -> native).

        Only the translational part scales with pixel size; rotation and scale
        factors are dimensionless and unchanged.
        """
        return Transform(
            t1=self.t1 * factor,
            t2=self.t2 * factor,
            rot=self.rot,
            s1=self.s1,
            s2=self.s2,
            score=self.score,
        )

    def __repr__(self) -> str:
        return (
            f"Transform(t=({self.t1:.3f}, {self.t2:.3f}), rot={self.rot:.5f} rad, "
            f"s=({self.s1:.5f}, {self.s2:.5f}), score={self.score:.4f})"
        )
