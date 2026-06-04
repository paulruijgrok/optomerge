"""
optomerge_oo.pipeline
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
        Channel layout; defaults to auto-detect.
    aligner : Aligner, optional
        Registration strategy; defaults to phase correlation.
    bunch_size : int
        Frames per processing block (the FrameBunch granularity).
    bg_radius : int
        Background-subtraction structuring-element radius.
    projection_frames : int, optional
        Frames used for channel finding / alignment (all if None).
    verbose : bool
    """

    def __init__(
        self,
        source: str | Path,
        layout: Optional[ChannelLayout] = None,
        aligner: Optional[Aligner] = None,
        bunch_size: int = 100_000,
        bg_radius: int = 10,
        projection_frames: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.source = Path(source)
        self.layout = layout or ChannelLayout.auto()
        self.aligner = aligner or PhaseCorrelationAligner()
        self.bunch_size = bunch_size
        self.bg_radius = bg_radius
        self.projection_frames = projection_frames
        self.verbose = verbose

        # Populated by run().
        self.resolved_layout: Optional[ChannelLayout] = None
        self.transforms: Dict[str, Transform] = {}
        self._limits: Dict[str, Tuple[float, float]] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # -- stages ------------------------------------------------------------ #

    def calibrate(self, movie: RawMovie) -> None:
        """Resolve the layout and fit per-moving-channel transforms.

        Mirrors the original ``find_channels`` + ``compute_alignment``: detect
        channel bounds on the projections, then phase-correlate the normalised
        projection crops. Intensity limits measured here are reused verbatim for
        per-frame normalisation (a key fidelity point).
        """
        self.resolved_layout = movie.find_layout(
            self.layout, projection_frames=self.projection_frames, verbose=self.verbose
        )
        self._log(f"Resolved layout: {self.resolved_layout.name} "
                  f"({len(self.resolved_layout.specs)} channel(s))")

        mean_proj = movie.mean_projection(self.projection_frames)
        proj_channels = {c.name: c.calibrate() for c in self.resolved_layout.split(mean_proj)}
        self._limits = {name: (c.vmin, c.vmax) for name, c in proj_channels.items()}

        reference = next(c for c in proj_channels.values() if c.reference)
        for spec in self.resolved_layout.moving_specs:
            self.transforms[spec.name] = self.aligner.align(reference, proj_channels[spec.name])
            self._log(f"  {spec.name}: {self.transforms[spec.name]}")

    def run(
        self,
        output: Optional[str | Path] = None,
        blue_led_frames: Optional[np.ndarray] = None,
    ) -> RGBMovie:
        """Execute the full merge and (optionally) write the result."""
        movie = RawMovie.open(self.source)
        self.calibrate(movie)

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
