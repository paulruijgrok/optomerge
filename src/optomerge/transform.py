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
            return np.asarray(image, dtype=np.float64)
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
