"""
optomerge_oo.movie
================
Movie hierarchy: Movie (abstract) -> RawMovie, RGBMovie.

  Movie     -- common behaviour for any ordered frame stack: shape, lazy frame
               access, projections, iteration over FrameBunches, save().
  RawMovie  -- a Movie read from disk whose frames still carry all channels
               spatially packed; knows how to find/split channels.
  RGBMovie  -- a Movie produced by merging aligned channels into false colour;
               knows the output-only concerns (colour mapping, crop range,
               dtype scaling for writing).
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

import numpy as np

from ._core import norm_image, subtract_background, zero_pad_images
from .io_backends import MovieReader, MovieWriter

if TYPE_CHECKING:
    from .channel import Channel
    from .frame import Frame, FrameBunch
    from .layout import ChannelLayout
    from .transform import Transform

_COLOR_INDEX = {"red": 0, "green": 1, "blue": 2}


class Movie(ABC):
    """Abstract ordered stack of frames with pixel metadata.

    A Movie does not assume its pixels are in RAM: it may hold a
    :class:`MovieReader` and pull data on demand. Concrete subclasses decide how
    a frame is *interpreted*, not how it is *stored*.

    Parameters
    ----------
    reader : MovieReader, optional
        Lazy data source. Either ``reader`` or ``data`` must be given.
    data : np.ndarray, optional
        In-memory array for an already-realised movie.
    pixel_size : float, optional
        Physical pixel size (microns), carried through for downstream analysis.
    """

    def __init__(
        self,
        reader: Optional[MovieReader] = None,
        data: Optional[np.ndarray] = None,
        pixel_size: Optional[float] = None,
    ) -> None:
        if reader is None and data is None:
            raise ValueError("Movie needs either a reader or data.")
        self._reader = reader
        self._data = None if data is None else np.asarray(data, dtype=np.float64)
        self.pixel_size = pixel_size

    # -- shape / access ---------------------------------------------------- #

    @property
    def shape(self) -> tuple:
        if self._data is not None:
            return self._data.shape
        return self._reader.shape()

    def __len__(self) -> int:
        return self.shape[-1]

    def to_array(self) -> np.ndarray:
        """Realise the whole movie in memory (loads if backed by a reader)."""
        if self._data is None:
            bunches = list(self.bunches(bunch_size=10_000))
            self._data = np.concatenate([b.data for b in bunches], axis=2)
        return self._data

    def frame(self, i: int) -> "Frame":
        from .frame import Frame
        if self._data is not None:
            return Frame(self._data[:, :, i], index=i)
        return self._reader.read_bunch(i, 1).frame(0)

    def bunches(self, bunch_size: int) -> "Iterator[FrameBunch]":
        from .frame import FrameBunch
        if self._reader is not None:
            yield from self._reader.iter_bunches(bunch_size)
            return
        n = self._data.shape[2]
        start = 0
        while start < n:
            count = min(bunch_size, n - start)
            yield FrameBunch(self._data[:, :, start:start + count], start_index=start)
            start += count

    # -- reductions -------------------------------------------------------- #

    def max_projection(self, n: Optional[int] = None) -> np.ndarray:
        if self._data is not None:
            sub = self._data if n is None else self._data[:, :, :n]
            return sub.max(axis=2)
        return self._streamed_projection("max", n)

    def mean_projection(self, n: Optional[int] = None) -> np.ndarray:
        if self._data is not None:
            sub = self._data if n is None else self._data[:, :, :n]
            return sub.mean(axis=2)
        return self._streamed_projection("mean", n)

    def _streamed_projection(self, how: str, n: Optional[int]) -> np.ndarray:
        acc = None
        count = 0
        for bunch in self.bunches(bunch_size=200):
            block = bunch.data
            if n is not None and count + block.shape[2] > n:
                block = block[:, :, : n - count]
            if how == "max":
                bproj = block.max(axis=2)
                acc = bproj if acc is None else np.maximum(acc, bproj)
            else:  # mean -> accumulate sum
                bsum = block.sum(axis=2)
                acc = bsum if acc is None else acc + bsum
            count += block.shape[2]
            if n is not None and count >= n:
                break
        if how == "mean":
            acc = acc / max(count, 1)
        return acc

    # -- output ------------------------------------------------------------ #

    def save(self, path: str | Path, writer: Optional[MovieWriter] = None) -> None:
        """Serialise via an explicit writer, or one chosen by extension."""
        writer = writer or MovieWriter.for_path(path)
        writer.write(self, path)


class RawMovie(Movie):
    """A movie straight off disk: channels are still spatially packed per frame."""

    @classmethod
    def open(cls, path: str | Path, pixel_size: Optional[float] = None) -> "RawMovie":
        """Open a raw movie lazily, picking a reader by file extension."""
        return cls(reader=MovieReader.for_path(path), pixel_size=pixel_size)

    def find_layout(
        self,
        candidate: "ChannelLayout",
        projection_frames: Optional[int] = None,
        verbose: bool = False,
    ) -> "ChannelLayout":
        """Resolve a layout against this movie's projections.

        Wraps segmentation; for a manual layout it still runs ``resolve`` so the
        bounds/scrub masks are produced in the canonical format.
        """
        max_proj = self.max_projection(projection_frames)
        mean_proj = self.mean_projection(projection_frames)
        return candidate.resolve(max_proj, mean_proj, verbose=verbose)

    def channels(
        self, layout: "ChannelLayout", bunch: Optional["FrameBunch"] = None
    ) -> "List[Channel]":
        """Split this movie (or one bunch) into channels per the layout."""
        source = bunch.data if bunch is not None else self.to_array()
        return layout.split(source)


class RGBMovie(Movie):
    """An aligned, merged, false-coloured output movie.

    Parameters
    ----------
    data : np.ndarray, shape (H, W, 3, N)
        Float RGB movie in [0, 1].
    crop_range : tuple of int, optional
        ``(rowbeg, rowend, colbeg, colend)`` shared across bunches.
    """

    def __init__(
        self,
        data: np.ndarray,
        crop_range: Optional[tuple] = None,
        pixel_size: Optional[float] = None,
    ) -> None:
        super().__init__(data=data, pixel_size=pixel_size)
        self.crop_range = crop_range

    # to_array on an RGBMovie returns the (H,W,3,N) array directly
    def to_array(self) -> np.ndarray:
        return self._data

    @classmethod
    def from_channels(
        cls,
        channels: "List[Channel]",
        transforms: "Dict[str, Transform]",
        crop_range: Optional[tuple] = None,
        bg_radius: int = 10,
        threshold: float = 0.01,
        n_colors: int = 3,
        blue_led_frames: Optional[np.ndarray] = None,
        pixel_size: Optional[float] = None,
    ) -> "RGBMovie":
        """Assemble an output movie from aligned channels.

        OO home of ``_align_frames_with_red`` / ``_align_frames_no_red``. Because
        it consumes a *list* of channels and a *dict* of transforms, the
        1-channel and 2-channel cases share one code path (and N>2 is a natural
        extension via the generalised padding branch).

        Faithfulness: with exactly one reference + one moving channel this
        reproduces the original pipeline arithmetic (normalise -> reference bg
        subtract -> zero-pad pair -> transform moving -> moving bg subtract ->
        intersect-threshold crop -> place in colour planes).
        """
        reference = next((c for c in channels if c.reference), channels[0])
        moving = [c for c in channels if not c.reference]

        # Reference: normalise then background-subtract.
        ref_arr = subtract_background(reference.normalized(), radius=bg_radius)

        prepared: List[tuple] = []  # (color, array) for each moving channel
        masks_for_crop = [ref_arr]

        if len(moving) == 1:
            mov = moving[0]
            mov_arr = mov.normalized()
            # Zero-pad the moving/reference pair exactly as the original code.
            mov_arr, ref_arr = zero_pad_images(mov_arr, ref_arr)
            mov_arr = transforms[mov.name].apply(mov_arr)
            mov_arr = subtract_background(mov_arr, radius=bg_radius)
            prepared.append((mov.color, mov_arr))
            masks_for_crop = [ref_arr, mov_arr]
        elif len(moving) > 1:
            # Generalised path: pad all channels to a common power-of-2 size.
            arrays = [ref_arr] + [m.normalized() for m in moving]
            arrays = _zero_pad_many(arrays)
            ref_arr = arrays[0]
            for mov, marr in zip(moving, arrays[1:]):
                marr = transforms[mov.name].apply(marr)
                marr = subtract_background(marr, radius=bg_radius)
                prepared.append((mov.color, marr))
            masks_for_crop = [ref_arr] + [a for _, a in prepared]

        # -- crop range (shared across bunches) --
        if crop_range is None:
            crop_range = _intersect_crop(masks_for_crop, threshold)
        rb, re, cb, ce = crop_range

        ref_crop = ref_arr[rb:re + 1, cb:ce + 1, :]
        Hc, Wc, num_frames = ref_crop.shape

        rgb = np.zeros((Hc, Wc, n_colors, num_frames), dtype=np.float64)
        rgb[:, :, _COLOR_INDEX[reference.color], :] = ref_crop
        for color, arr in prepared:
            rgb[:, :, _COLOR_INDEX[color], :] = arr[rb:re + 1, cb:ce + 1, :]

        # Blue LED plane.
        if blue_led_frames is not None and "blue" in _COLOR_INDEX and n_colors > _COLOR_INDEX["blue"]:
            flags = np.asarray(blue_led_frames, dtype=bool)
            for i in range(num_frames):
                if i < len(flags) and flags[i]:
                    rgb[:, :, _COLOR_INDEX["blue"], i] = 0.5

        return cls(rgb, crop_range=crop_range, pixel_size=pixel_size)


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #

def _intersect_crop(arrays: List[np.ndarray], threshold: float) -> tuple:
    """Tightest bounding box where *all* mean-projected arrays exceed threshold."""
    cropping = None
    for arr in arrays:
        mask = arr.mean(axis=2) > threshold
        cropping = mask if cropping is None else (cropping & mask)
    croprows = cropping.any(axis=1)
    cropcols = cropping.any(axis=0)
    rowbeg = int(np.argmax(croprows))
    rowend = int(len(croprows) - 1 - np.argmax(croprows[::-1]))
    colbeg = int(np.argmax(cropcols))
    colend = int(len(cropcols) - 1 - np.argmax(cropcols[::-1]))
    return rowbeg, rowend, colbeg, colend


def _zero_pad_many(arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Centre-pad N stacks to a common power-of-2 (H, W); generalises zero_pad_images."""
    from optomerge.transform import _next_pow2  # reuse the exact rule

    n = arrays[0].shape[2]
    new_H = _next_pow2(max(a.shape[0] for a in arrays))
    new_W = _next_pow2(max(a.shape[1] for a in arrays))
    out = []
    for a in arrays:
        H, W, _ = a.shape
        padded = np.zeros((new_H, new_W, n), dtype=np.float64)
        r = (new_H - H) // 2
        c = (new_W - W) // 2
        padded[r:r + H, c:c + W, :] = a
        out.append(padded)
    return out
