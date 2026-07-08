"""
optomerge.io_backends
======================
MovieReader / MovieWriter: pluggable I/O backends.

This is the seam for "more types of input files" and "more output formats". A
reader exposes frames lazily (bunch by bunch, so a movie larger than RAM can be
streamed); a writer consumes a Movie and serialises it. New formats -- ND2, CZI,
HDF5, OME-Zarr, MP4 -- are new subclasses registered by file extension, with no
changes to the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, Type

from ._core import get_movie_info, load_frames, save_rgb_tiff

if TYPE_CHECKING:
    from .frame import FrameBunch
    from .movie import Movie


class MovieReader(ABC):
    """Lazy, format-specific source of frames."""

    # extension -> subclass, populated by register()
    _registry: Dict[str, Type["MovieReader"]] = {}

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @abstractmethod
    def shape(self) -> tuple[int, int, int]:
        """Return ``(H, W, N)`` without reading pixel data."""

    @abstractmethod
    def read_bunch(self, start: int, count: int) -> "FrameBunch":
        """Read ``count`` frames starting at ``start`` (0-indexed)."""

    def iter_bunches(self, bunch_size: int) -> "Iterator[FrameBunch]":
        """Yield consecutive FrameBunches covering the whole movie."""
        _, _, n = self.shape()
        start = 0
        while start < n:
            count = min(bunch_size, n - start)
            yield self.read_bunch(start, count)
            start += count

    # -- registry dispatch ------------------------------------------------- #

    @classmethod
    def register(cls, *extensions: str):
        def _wrap(subclass: Type["MovieReader"]) -> Type["MovieReader"]:
            for ext in extensions:
                cls._registry[ext.lower()] = subclass
            return subclass
        return _wrap

    @classmethod
    def for_path(cls, path: str | Path) -> "MovieReader":
        ext = Path(path).suffix.lower()
        if ext not in cls._registry:
            raise ValueError(
                f"No MovieReader registered for '{ext}'. "
                f"Known: {sorted(cls._registry)}"
            )
        return cls._registry[ext](path)


class MovieWriter(ABC):
    """Format-specific sink for a processed movie."""

    _registry: Dict[str, Type["MovieWriter"]] = {}

    @abstractmethod
    def write(self, movie: "Movie", path: str | Path) -> None:
        """Serialise ``movie`` to ``path``."""

    @classmethod
    def register(cls, *extensions: str):
        def _wrap(subclass: Type["MovieWriter"]) -> Type["MovieWriter"]:
            for ext in extensions:
                cls._registry[ext.lower()] = subclass
            return subclass
        return _wrap

    @classmethod
    def for_path(cls, path: str | Path, **kwargs) -> "MovieWriter":
        ext = Path(path).suffix.lower()
        if ext not in cls._registry:
            raise ValueError(
                f"No MovieWriter registered for '{ext}'. "
                f"Known: {sorted(cls._registry)}"
            )
        return cls._registry[ext](**kwargs)


@MovieReader.register(".tif", ".tiff")
class TiffReader(MovieReader):
    """Multi-page TIFF reader (wraps ``optomerge.io.load_frames``)."""

    def shape(self) -> tuple[int, int, int]:
        return get_movie_info(self.path)

    def read_bunch(self, start: int, count: int) -> "FrameBunch":
        from .frame import FrameBunch
        data = load_frames(self.path, start, start + count - 1)
        return FrameBunch(data, start_index=start)


@MovieWriter.register(".tif", ".tiff")
class TiffWriter(MovieWriter):
    """Colour multi-page TIFF writer (wraps ``optomerge.io.save_rgb_tiff``).

    ``bit_depth`` is 8 (default, compact uint8 RGB) or 16.
    """

    def __init__(self, bit_depth: int = 8) -> None:
        self.bit_depth = bit_depth

    def write(self, movie: "Movie", path: str | Path) -> None:
        arr = movie.to_array()
        if arr.ndim != 4 or arr.shape[2] != 3:
            raise ValueError(
                f"TiffWriter expects an (H, W, 3, N) RGB movie, got {arr.shape}."
            )
        save_rgb_tiff(path, arr, bit_depth=self.bit_depth)
