"""
optomerge_oo.layout
==================
ChannelSpec and ChannelLayout: declarative description of how channels are
packed into a raw frame, and which output colour each maps to.

This is the seam for the "more combinations of channels to be cropped and
aligned" goal. The two channel orders the original code hard-codes become two
factory calls; supporting 3-up, left/right, or quad views is adding a layout
instance, not editing branches in the pipeline.

A layout starts *unresolved* (bounds unknown). :meth:`ChannelLayout.resolve`
runs the segmentation on projection images and returns a *resolved* copy whose
:class:`ChannelSpec`s carry concrete bounds and scrub masks; that resolved
layout can then :meth:`split` raw pixel data into :class:`Channel`s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from ._core import crop_channel, find_channel_bounds
from .channel import Channel

if TYPE_CHECKING:
    pass


@dataclass
class ChannelSpec:
    """Declarative description of one channel within a raw frame.

    Parameters
    ----------
    name : str
        Logical name, e.g. ``"green"``.
    color : str
        Output colour: ``"red"`` / ``"green"`` / ``"blue"``.
    bounds : np.ndarray, optional
        Crop ``[[row_start, row_end], [col_start, col_end]]`` (0-indexed,
        inclusive). ``None`` until resolved by segmentation.
    mask : np.ndarray, optional
        Full-frame boolean scrub mask. ``None`` until resolved.
    reference : bool
        Whether this channel is the alignment reference.
    """

    name: str
    color: str
    bounds: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    reference: bool = False

    @property
    def resolved(self) -> bool:
        return self.bounds is not None and self.mask is not None


@dataclass
class ChannelLayout:
    """How a raw frame is partitioned into channels.

    Parameters
    ----------
    specs : list of ChannelSpec
        Populated after :meth:`resolve` (empty for an unresolved auto layout).
    name : str
        Human-readable layout id.
    channel_order : str
        Passed through to the segmentation: ``"auto"``,
        ``"top_green_fils_bottom_red_heads"`` or
        ``"top_red_heads_bottom_green_fils"``.
    manual_bounds : np.ndarray, optional
        Manual bounds ``[x_left, y_top, x_right, y_bottom]`` per channel; when
        given the line-search is skipped.
    """

    specs: List[ChannelSpec] = field(default_factory=list)
    name: str = "auto"
    channel_order: str = "auto"
    manual_bounds: Optional[np.ndarray] = None

    # -- queries ----------------------------------------------------------- #

    @property
    def is_resolved(self) -> bool:
        return len(self.specs) > 0 and all(s.resolved for s in self.specs)

    @property
    def is_single_channel(self) -> bool:
        return len(self.specs) == 1

    @property
    def reference_spec(self) -> ChannelSpec:
        for s in self.specs:
            if s.reference:
                return s
        if self.specs:
            return self.specs[0]
        raise RuntimeError("Layout has no specs; call resolve() first.")

    @property
    def moving_specs(self) -> List[ChannelSpec]:
        return [s for s in self.specs if not s.reference]

    # -- resolution -------------------------------------------------------- #

    def resolve(
        self,
        max_projection: np.ndarray,
        mean_projection: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "ChannelLayout":
        """Run segmentation and return a resolved copy with concrete bounds.

        Wraps the functional ``find_channel_bounds``. Channel 1 (bright) is the
        green reference; channel 2 (if present) is the red moving channel.
        """
        b1, s1, b2, s2 = find_channel_bounds(
            max_projection,
            mean_projection=mean_projection,
            channel_order=self.channel_order,
            channel_bounds=self.manual_bounds,
            verbose=verbose,
        )

        specs = [ChannelSpec(name="green", color="green", bounds=b1, mask=s1, reference=True)]
        if b2 is not None:
            specs.append(ChannelSpec(name="red", color="red", bounds=b2, mask=s2, reference=False))

        return ChannelLayout(
            specs=specs,
            name=self.name,
            channel_order=self.channel_order,
            manual_bounds=self.manual_bounds,
        )

    # -- splitting --------------------------------------------------------- #

    def split(self, image: np.ndarray) -> List[Channel]:
        """Crop raw pixel data ``(H, W)`` or ``(H, W, N)`` into Channels."""
        if not self.is_resolved:
            raise RuntimeError("Cannot split with an unresolved layout; call resolve() first.")
        channels: List[Channel] = []
        for spec in self.specs:
            cropped = crop_channel(image, spec.bounds, spec.mask)
            channels.append(
                Channel(
                    cropped, name=spec.name, bounds=spec.bounds,
                    color=spec.color, mask=spec.mask, reference=spec.reference,
                )
            )
        return channels

    # -- factories --------------------------------------------------------- #

    @classmethod
    def auto(cls) -> "ChannelLayout":
        """Layout whose bounds and channel count are discovered at runtime."""
        return cls(name="auto", channel_order="auto")

    @classmethod
    def top_green_bottom_red(cls) -> "ChannelLayout":
        """Equivalent of ``top_green_fils_bottom_red_heads``."""
        return cls(name="top_green_bottom_red",
                   channel_order="top_green_fils_bottom_red_heads")

    @classmethod
    def top_red_bottom_green(cls) -> "ChannelLayout":
        """Equivalent of ``top_red_heads_bottom_green_fils``."""
        return cls(name="top_red_bottom_green",
                   channel_order="top_red_heads_bottom_green_fils")

    @classmethod
    def manual(cls, bounds: np.ndarray, channel_order: str = "auto") -> "ChannelLayout":
        """Layout with explicit manual bounds (one row per channel)."""
        return cls(name="manual", channel_order=channel_order,
                   manual_bounds=np.asarray(bounds))
