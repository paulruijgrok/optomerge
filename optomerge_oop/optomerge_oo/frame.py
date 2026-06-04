"""
optomerge_oo.frame
==================
Frame and FrameBunch: the unit of pixel data and the unit of work.

A *Movie* is a logical, possibly out-of-core stack. A *Frame* is one realised
2-D image, and a *FrameBunch* is the chunk that actually flows through the
processing/registration code. Decoupling "the whole movie" from "the block I am
currently crunching" is what lets the package scale: the FrameBunch is where a
future chunked / streaming / GPU executor plugs in, without Movie or the
pipeline having to know how many frames fit in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from .channel import Channel
    from .layout import ChannelLayout


@dataclass
class Frame:
    """A single 2-D image within a movie.

    Parameters
    ----------
    data : np.ndarray, shape (H, W)
        Pixel data for this frame (raw camera frame, channels still packed).
    index : int
        Position of this frame in its parent movie (0-indexed).
    time : float, optional
        Acquisition timestamp in seconds, if known.
    metadata : dict, optional
        Per-frame metadata, e.g. ``{"blue_led": True}``.
    """

    data: np.ndarray
    index: int
    time: Optional[float] = None
    metadata: Optional[dict] = None

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def crop(self, layout: "ChannelLayout", spec_name: str) -> "Channel":
        """Extract a single :class:`Channel` from this frame.

        Delegates geometry to ``layout`` so that *how* channels are packed is
        not hard-coded here.
        """
        for ch in layout.split(self.data):
            if ch.name == spec_name:
                return ch
        raise KeyError(f"No channel named {spec_name!r} in layout.")

    def __array__(self) -> np.ndarray:
        return self.data


class FrameBunch:
    """A contiguous block of frames processed together as one unit of work.

    The bunch is chosen so that object motion across the block is small enough
    that a single projection is representative and the block fits a chosen
    memory / compute budget. It is the natural granularity for projections,
    vectorised transforms, and future parallel / out-of-core execution.

    Parameters
    ----------
    data : np.ndarray, shape (H, W, n)
        The stacked frames in this bunch.
    start_index : int
        Index of the first frame within the parent movie.
    metadata : sequence of dict, optional
        Per-frame metadata aligned with the frames in the bunch.
    """

    def __init__(
        self,
        data: np.ndarray,
        start_index: int,
        metadata: Optional[Sequence[dict]] = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        if self.data.ndim == 2:
            self.data = self.data[:, :, np.newaxis]
        self.start_index = start_index
        self.metadata = metadata

    # -- size / iteration -------------------------------------------------- #

    def __len__(self) -> int:
        return self.data.shape[2]

    def frame(self, i: int) -> Frame:
        meta = self.metadata[i] if self.metadata is not None else None
        return Frame(self.data[:, :, i], index=self.start_index + i, metadata=meta)

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self.frame(i)

    # -- reductions used by channel-finding / alignment -------------------- #

    def max_projection(self) -> np.ndarray:
        return self.data.max(axis=2)

    def mean_projection(self) -> np.ndarray:
        return self.data.mean(axis=2)

    # -- channel extraction ------------------------------------------------ #

    def channels(self, layout: "ChannelLayout") -> "List[Channel]":
        """Split every frame in the bunch into channels -> stacked Channels."""
        return layout.split(self.data)

    def extract_channel(self, layout: "ChannelLayout", spec_name: str) -> "Channel":
        """Crop one named channel from every frame in the bunch."""
        for ch in self.channels(layout):
            if ch.name == spec_name:
                return ch
        raise KeyError(f"No channel named {spec_name!r} in layout.")
