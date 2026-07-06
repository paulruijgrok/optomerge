"""
optomerge.calibration
======================
Calibrators: strategies that turn a raw movie into a *calibration* — the
resolved channel layout, the fitted per-channel transforms, and the intensity
limits used for normalisation — plus an acceptance score.

This is the seam for "more robust channel/alignment finding". Two strategies
ship:

* :class:`SingleProjectionCalibrator` — project the whole movie once, detect
  channels, fit the alignment. Fast; the original behaviour.
* :class:`RobustCalibrator` — the port of ``findChannelsAndAlignment`` from
  ``alignRGB.m``: project the movie in chunks, fit + score each, keep only
  candidates that pass the :mod:`~optomerge.acceptance` criteria, and return the
  best of N. Robust to a bad stretch of frames (drift, blur, cosmic rays).

Both return a :class:`Calibration`, so :class:`~optomerge.pipeline.MergePipeline`
consumes them through one interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .acceptance import AcceptanceCriteria, evaluate
from .layout import ChannelLayout
from .registration import Aligner, PhaseCorrelationAligner

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .movie import RawMovie
    from .transform import Transform


class AlignmentNotFoundError(RuntimeError):
    """Raised when no candidate alignment passes the acceptance criteria."""


@dataclass
class Calibration:
    """Result of calibrating a movie: layout + transforms + limits (+ score)."""

    resolved_layout: ChannelLayout
    transforms: Dict[str, "Transform"]
    limits: Dict[str, Tuple[float, float]]
    score: float = float("nan")
    accepted: bool = True
    reasons: List[str] = field(default_factory=list)
    image_shape: Optional[tuple] = None


def _despeckle(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Spatial median filter to remove isolated cosmic rays / hot pixels.

    Unlike a percentile clip, a 3x3 median removes lone bright pixels but keeps
    multi-pixel features (e.g. sparse point-like "heads"), so ``vmax`` reflects
    real signal rather than being pulled down by clipping sparse channels.
    """
    try:
        from scipy.ndimage import median_filter
        return median_filter(np.asarray(image, dtype=np.float64), size=size)
    except Exception:  # pragma: no cover - scipy always present in practice
        return np.asarray(image, dtype=np.float64)


def _compute_limits(
    resolved_layout: ChannelLayout,
    limit_source: np.ndarray,
    norm_exclude: float,
) -> Dict[str, Tuple[float, float]]:
    """Per-channel (vmin, vmax) intensity limits from a projection.

    The projection is median-filtered first (removes cosmic rays without
    clipping sparse real features), then ``vmin``/``vmax`` are the min/max --
    or, if ``norm_exclude > 0``, the corresponding percentiles.
    """
    despeckled = _despeckle(limit_source)
    chans = {c.name: c.calibrate(exclude_fraction=norm_exclude)
             for c in resolved_layout.split(despeckled)}
    return {name: (c.vmin, c.vmax) for name, c in chans.items()}


def _build_calibration(
    resolved_layout: ChannelLayout,
    max_proj: np.ndarray,
    mean_proj: np.ndarray,
    aligner: Aligner,
    criteria: Optional[AcceptanceCriteria],
    gate: bool,
    norm_from_max: bool = True,
    norm_exclude: float = 0.0,
) -> Calibration:
    """Fit transforms and package a :class:`Calibration`.

    Normalisation limits are taken from the **max** projection (``norm_from_max``)
    so per-frame filament peaks are not clipped -- the mean projection blurs
    moving objects and yields too-low a ``vmax``, saturating the output. The
    *alignment* still uses the mean projection (smoother, better registration).

    ``gate`` controls whether acceptance failure sets ``accepted = False``
    (robust filtering) or is only recorded as a score (single-projection).
    """
    limit_source = max_proj if norm_from_max else mean_proj
    limits = _compute_limits(resolved_layout, limit_source, norm_exclude)

    align_channels = {c.name: c.calibrate() for c in resolved_layout.split(mean_proj)}
    reference = next(c for c in align_channels.values() if c.reference)

    transforms: Dict[str, "Transform"] = {}
    for spec in resolved_layout.moving_specs:
        transforms[spec.name] = aligner.align(reference, align_channels[spec.name])

    result = evaluate(resolved_layout.specs, transforms, mean_proj.shape,
                      criteria or AcceptanceCriteria())
    return Calibration(
        resolved_layout=resolved_layout,
        transforms=transforms,
        limits=limits,
        score=result.score,
        accepted=result.passed if gate else True,
        reasons=result.reasons if gate else [],
        image_shape=tuple(mean_proj.shape),
    )


class Calibrator(ABC):
    """Strategy that produces a :class:`Calibration` from a raw movie."""

    @abstractmethod
    def calibrate(self, movie: "RawMovie") -> Calibration:
        """Detect channels and fit the alignment for ``movie``."""


class SingleProjectionCalibrator(Calibrator):
    """Detect + align from a single projection of the whole movie (or N frames).

    Parameters
    ----------
    layout : ChannelLayout, optional
        Candidate layout (defaults to auto-detect).
    aligner : Aligner, optional
        Registration strategy (defaults to phase correlation).
    projection_frames : int, optional
        Frames used for the detection projection (all if None).
    criteria : AcceptanceCriteria, optional
        If given, the returned calibration's ``accepted`` reflects these; a
        single-projection result is still returned (not raised) so the caller
        can decide what to do with a rejected fit.
    verbose : bool
    """

    def __init__(
        self,
        layout: Optional[ChannelLayout] = None,
        aligner: Optional[Aligner] = None,
        projection_frames: Optional[int] = None,
        criteria: Optional[AcceptanceCriteria] = None,
        verbose: bool = False,
        norm_from_max: bool = True,
        norm_exclude: float = 0.0,
    ) -> None:
        self.layout = layout or ChannelLayout.auto()
        self.aligner = aligner or PhaseCorrelationAligner()
        self.projection_frames = projection_frames
        self.criteria = criteria
        self.verbose = verbose
        self.norm_from_max = norm_from_max
        self.norm_exclude = norm_exclude

    def calibrate(self, movie: "RawMovie") -> Calibration:
        max_proj = movie.max_projection(self.projection_frames)
        mean_proj = movie.mean_projection(self.projection_frames)
        resolved = self.layout.resolve(max_proj, mean_proj, verbose=self.verbose)
        return _build_calibration(resolved, max_proj, mean_proj, self.aligner,
                                  self.criteria, gate=self.criteria is not None,
                                  norm_from_max=self.norm_from_max,
                                  norm_exclude=self.norm_exclude)


class ConservedCalibrator(Calibrator):
    """Reuse a reference :class:`Calibration` across movies in a set.

    Applies the reference layout + transforms verbatim but recomputes the
    intensity limits from *this* movie's projection, so each movie is normalised
    to its own range while sharing the (virtually identical) alignment. This is
    the OO home of ``alignRGB.m``'s ``keepAlignment`` behaviour.
    """

    def __init__(self, reference: Calibration, projection_frames: Optional[int] = None,
                 norm_from_max: bool = True, norm_exclude: float = 0.0) -> None:
        self.reference = reference
        self.projection_frames = projection_frames
        self.norm_from_max = norm_from_max
        self.norm_exclude = norm_exclude

    def calibrate(self, movie: "RawMovie") -> Calibration:
        layout = self.reference.resolved_layout
        limit_source = (movie.max_projection(self.projection_frames) if self.norm_from_max
                        else movie.mean_projection(self.projection_frames))
        limits = _compute_limits(layout, limit_source, self.norm_exclude)
        return Calibration(
            resolved_layout=layout,
            transforms=dict(self.reference.transforms),
            limits=limits,
            score=self.reference.score,
            accepted=True,
            reasons=[],
            image_shape=tuple(limit_source.shape),
        )


class RobustCalibrator(Calibrator):
    """Best-of-N calibrator: chunk the movie, score each, pick the best.

    Port of ``findChannelsAndAlignment`` (``alignRGB.m``). The movie is projected
    in ``chunk_size``-frame blocks; each block is detected + aligned + scored,
    and only blocks passing the acceptance criteria are kept. Once
    ``min_candidates`` have passed (or ``max_trials`` blocks have been tried) the
    highest-scoring candidate is returned. Robust to a bad stretch of frames.

    Parameters
    ----------
    layout, aligner : as :class:`SingleProjectionCalibrator`.
    criteria : AcceptanceCriteria, optional
        Acceptance thresholds (defaults to :class:`AcceptanceCriteria`).
    chunk_size : int
        Frames per projection block (``projectionChunkSize``, default 400).
    min_candidates : int
        Stop once this many blocks pass acceptance (``minNumTopAlignments``).
    max_trials : int
        Give up after this many blocks (``maxNumTrials``).
    verbose : bool
    """

    def __init__(
        self,
        layout: Optional[ChannelLayout] = None,
        aligner: Optional[Aligner] = None,
        criteria: Optional[AcceptanceCriteria] = None,
        chunk_size: int = 400,
        min_candidates: int = 10,
        max_trials: int = 30,
        verbose: bool = False,
        norm_from_max: bool = True,
        norm_exclude: float = 0.0,
    ) -> None:
        self.layout = layout or ChannelLayout.auto()
        self.aligner = aligner or PhaseCorrelationAligner()
        self.criteria = criteria or AcceptanceCriteria()
        self.chunk_size = chunk_size
        self.min_candidates = min_candidates
        self.max_trials = max_trials
        self.verbose = verbose
        self.norm_from_max = norm_from_max
        self.norm_exclude = norm_exclude

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def calibrate(self, movie: "RawMovie") -> Calibration:
        n_frames = movie.shape[-1]

        # Too few frames to chunk: fall back to a single projection of all frames.
        if n_frames <= self.chunk_size:
            self._log(f"RobustCalibrator: {n_frames} frames <= chunk {self.chunk_size}; "
                      "single projection.")
            single = SingleProjectionCalibrator(
                self.layout, self.aligner, projection_frames=None,
                criteria=self.criteria, verbose=self.verbose,
                norm_from_max=self.norm_from_max, norm_exclude=self.norm_exclude,
            ).calibrate(movie)
            if not single.accepted:
                raise AlignmentNotFoundError(
                    "single-projection alignment did not pass acceptance: "
                    + "; ".join(single.reasons)
                )
            return single

        candidates: List[Calibration] = []
        best_rejected: Optional[Calibration] = None
        trials = 0

        for bunch in movie.bunches(self.chunk_size):
            if len(bunch) < self.chunk_size:
                continue  # skip a short trailing block
            trials += 1
            try:
                max_proj = bunch.max_projection()
                mean_proj = bunch.mean_projection()
                resolved = self.layout.resolve(max_proj, mean_proj, verbose=False)
                cal = _build_calibration(resolved, max_proj, mean_proj, self.aligner,
                                         self.criteria, gate=True,
                                         norm_from_max=self.norm_from_max,
                                         norm_exclude=self.norm_exclude)
            except Exception as exc:  # a bad block shouldn't sink the movie
                self._log(f"  block @{bunch.start_index}: skipped ({exc})")
                continue  # counts as a trial (guards against an all-bad movie)

            if cal.accepted and np.isfinite(cal.score):
                candidates.append(cal)
                self._log(f"  block @{bunch.start_index}: candidate "
                          f"{len(candidates)}/{self.min_candidates} score={cal.score:.4f}")
            else:
                if best_rejected is None or (np.isfinite(cal.score)
                                             and cal.score > best_rejected.score):
                    best_rejected = cal

            if len(candidates) >= self.min_candidates or trials >= self.max_trials:
                break

        if not candidates:
            reasons = best_rejected.reasons if best_rejected else ["no channels detected"]
            raise AlignmentNotFoundError(
                f"no acceptable alignment in {trials} block(s); "
                f"best rejected: {'; '.join(reasons)}"
            )

        best = max(candidates, key=lambda c: c.score)
        self._log(f"RobustCalibrator: best of {len(candidates)} candidate(s), "
                  f"score={best.score:.4f}")
        return best
