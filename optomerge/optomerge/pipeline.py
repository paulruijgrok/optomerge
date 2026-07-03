"""
optomerge.pipeline
===================
MergePipeline: the thin orchestrator.

The pipeline owns no algorithms. It wires together a RawMovie, a ChannelLayout,
an Aligner and a writer, and walks the movie bunch by bunch. All heavy logic
lives on the domain objects, so the pipeline reads like the high-level recipe
and stays format-/method-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .acceptance import AcceptanceCriteria
from .calibration import Calibration, Calibrator, SingleProjectionCalibrator
from .layout import ChannelLayout
from .movie import RGBMovie, RawMovie
from .registration import Aligner, PhaseCorrelationAligner
from .transform import Transform


class MergePipeline:
    """Orchestrates raw-movie -> aligned RGB movie.

    Parameters
    ----------
    source : str or Path
        Input movie path; a reader is selected by extension.
    layout : ChannelLayout, optional
        Channel layout; defaults to auto-detect. Ignored if ``calibrator`` is given.
    aligner : Aligner, optional
        Registration strategy; defaults to phase correlation. Ignored if
        ``calibrator`` is given.
    bunch_size : int
        Frames per processing block (the FrameBunch granularity).
    bg_radius : int
        Background-subtraction structuring-element radius.
    projection_frames : int, optional
        Frames used for channel finding / alignment (all if None). Ignored if
        ``calibrator`` is given.
    verbose : bool
    calibrator : Calibrator, optional
        Strategy for detecting channels + fitting the alignment. Defaults to a
        :class:`~optomerge.calibration.SingleProjectionCalibrator` built from
        ``layout``/``aligner``/``projection_frames``. Pass a
        :class:`~optomerge.calibration.RobustCalibrator` for best-of-N finding.
    criteria : AcceptanceCriteria, optional
        Acceptance thresholds; recorded on the resulting calibration (and, for
        the default single-projection calibrator, used to flag a rejected fit).
    """

    def __init__(
        self,
        source: str | Path,
        layout: Optional[ChannelLayout] = None,
        aligner: Optional[Aligner] = None,
        bunch_size: int = 100,
        bg_radius: int = 10,
        projection_frames: Optional[int] = None,
        verbose: bool = False,
        calibrator: Optional[Calibrator] = None,
        criteria: Optional[AcceptanceCriteria] = None,
    ) -> None:
        self.source = Path(source)
        self.layout = layout or ChannelLayout.auto()
        self.aligner = aligner or PhaseCorrelationAligner()
        self.bunch_size = bunch_size
        self.bg_radius = bg_radius
        self.projection_frames = projection_frames
        self.verbose = verbose
        self.criteria = criteria
        self.calibrator = calibrator or SingleProjectionCalibrator(
            layout=self.layout, aligner=self.aligner,
            projection_frames=projection_frames, criteria=criteria, verbose=verbose,
        )

        # Populated by run().
        self.calibration: Optional[Calibration] = None
        self.resolved_layout: Optional[ChannelLayout] = None
        self.transforms: Dict[str, Transform] = {}
        self._limits: Dict[str, Tuple[float, float]] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # -- stages ------------------------------------------------------------ #

    def calibrate(self, movie: RawMovie) -> None:
        """Detect the layout + fit transforms via the configured calibrator.

        Delegates to ``self.calibrator`` and stores the resulting layout,
        transforms and intensity limits. Intensity limits measured here are
        reused verbatim for per-frame normalisation (a key fidelity point).
        """
        cal = self.calibrator.calibrate(movie)
        self.calibration = cal
        self.resolved_layout = cal.resolved_layout
        self.transforms = cal.transforms
        self._limits = cal.limits
        self._log(f"Resolved layout: {cal.resolved_layout.name} "
                  f"({len(cal.resolved_layout.specs)} channel(s)); "
                  f"score={cal.score:.4f} accepted={cal.accepted}")
        for name, t in cal.transforms.items():
            self._log(f"  {name}: {t}")

    def run(
        self,
        output: Optional[str | Path] = None,
        blue_led_frames: Optional[np.ndarray] = None,
    ) -> RGBMovie:
        """Open the movie, calibrate, and merge (optionally writing the result).

        Convenience wrapper over :meth:`calibrate` + :meth:`merge`. Callers that
        need to inspect or gate on the calibration (e.g. reject a movie whose
        alignment failed acceptance) before the expensive merge should call the
        two stages separately.
        """
        movie = RawMovie.open(self.source)
        self.calibrate(movie)
        return self.merge(movie, output=output, blue_led_frames=blue_led_frames)

    def merge(
        self,
        movie: RawMovie,
        output: Optional[str | Path] = None,
        blue_led_frames: Optional[np.ndarray] = None,
    ) -> RGBMovie:
        """Assemble the aligned RGB movie using the current calibration.

        Requires :meth:`calibrate` (or a reused calibration) to have populated
        ``resolved_layout`` / ``transforms`` / ``_limits`` first.
        """
        if self.resolved_layout is None:
            raise RuntimeError("merge() called before calibrate(); no layout resolved.")

        if blue_led_frames is not None:
            blue_led_frames = np.asarray(blue_led_frames, dtype=bool)

        crop_range: Optional[Tuple] = None
        rgb_blocks = []

        for bunch in movie.bunches(self.bunch_size):
            channels = self.resolved_layout.split(bunch.data)
            for ch in channels:
                ch.vmin, ch.vmax = self._limits[ch.name]
                if ch.name in self.transforms:
                    ch.transform = self.transforms[ch.name]

            led_slice = None
            if blue_led_frames is not None:
                led_slice = blue_led_frames[bunch.start_index: bunch.start_index + len(bunch)]

            rgb = RGBMovie.from_channels(
                channels,
                self.transforms,
                crop_range=crop_range,
                bg_radius=self.bg_radius,
                blue_led_frames=led_slice,
                pixel_size=movie.pixel_size,
            )
            crop_range = rgb.crop_range  # lock crop range after the first bunch
            rgb_blocks.append(rgb.to_array())
            self._log(f"  processed frames {bunch.start_index}..{bunch.start_index + len(bunch) - 1}")

        merged = np.concatenate(rgb_blocks, axis=3)
        result = RGBMovie(merged, crop_range=crop_range, pixel_size=movie.pixel_size)

        if output is not None:
            result.save(output)
            self._log(f"Saved -> {output}")
        return result
