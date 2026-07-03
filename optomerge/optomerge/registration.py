"""
optomerge.registration
========================
Aligner: the registration strategy abstraction.

This is the seam for "more robust" and "more general" alignment. The current
phase-correlation method becomes one concrete Aligner; ECC, feature-based, or
optical-flow methods can be dropped in without touching the pipeline, because
they all return a :class:`Transform` from the same ``align`` call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ._core import calculate_alignment
from .transform import Transform

if TYPE_CHECKING:
    from .channel import Channel


class Aligner(ABC):
    """Strategy that computes a :class:`Transform` mapping ``moving`` -> ``reference``."""

    @abstractmethod
    def align(self, reference: "Channel", moving: "Channel") -> Transform:
        """Estimate the transform that registers ``moving`` onto ``reference``."""
        raise NotImplementedError


class PhaseCorrelationAligner(Aligner):
    """Port of the functional ``calculate_alignment`` (FFT phase correlation).

    By default it aligns on the normalised channel projections exactly as the
    original pipeline does, so results match frame-for-frame. Setting
    ``use_scrub=True`` aligns on upsampled :class:`ScrubImage`s and maps the
    fitted transform back to native pixels -- an opt-in accuracy path.

    Parameters
    ----------
    init_rot, init_s1, init_s2 : float
        Initial guesses for rotation / scale, as in the original code.
    use_scrub : bool
        Whether to align on upsampled ScrubImages for sub-pixel accuracy.
    upscale : int
        Scrub-image upscaling factor when ``use_scrub`` is True.
    """

    def __init__(
        self,
        init_rot: float = 0.0,
        init_s1: float = 1.0,
        init_s2: float = 1.0,
        use_scrub: bool = False,
        upscale: int = 4,
    ) -> None:
        self.init_rot = init_rot
        self.init_s1 = init_s1
        self.init_s2 = init_s2
        self.use_scrub = use_scrub
        self.upscale = upscale

    def align(self, reference: "Channel", moving: "Channel") -> Transform:
        if self.use_scrub:
            ref_img = reference.scrub_image(self.upscale).data
            mov_img = moving.scrub_image(self.upscale).data
        else:
            ref_img = reference.normalized()
            mov_img = moving.normalized()

        t1, t2, rot, s1, s2, score = calculate_alignment(
            ref_img, mov_img, self.init_rot, self.init_s1, self.init_s2
        )
        transform = Transform(t1=t1, t2=t2, rot=rot, s1=s1, s2=s2, score=score)

        if self.use_scrub:
            transform = transform.rescaled(1.0 / self.upscale)
        return transform
