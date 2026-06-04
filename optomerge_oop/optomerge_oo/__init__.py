"""
optomerge_oo
============
Object-oriented redesign of the optomerge package (skeleton / proposal).

This package is a *design proposal*. The method bodies are stubs that raise
``NotImplementedError`` or return placeholder values. The intent is to show how
the domain concepts map onto classes and how the boundaries between them keep
the package extensible toward the three stated goals:

  * faster   -> swap eager numpy for lazy/chunked/GPU backends behind one API
  * robust   -> validation lives on the objects that own the data
  * general  -> readers, channel layouts, aligners and writers are pluggable

Core domain objects (the ones the user sketched)
------------------------------------------------
  Movie       -- abstract base: an ordered stack of frames + pixel metadata
  RawMovie    -- a Movie straight off disk; channels are spatially packed
  RGBMovie    -- a Movie whose frames are merged, aligned, false-coloured output
  Frame       -- a single 2-D image within a movie (+ its index/time)
  FrameBunch  -- a contiguous block of frames processed together
  Channel     -- one cropped sub-region of a raw frame (one molecular species)
  ScrubImage  -- an enlarged/upsampled channel used to drive sub-pixel alignment

Supporting abstractions (the seams that make extension cheap)
-------------------------------------------------------------
  MovieReader / MovieWriter -- I/O backends (TIFF, ND2, HDF5, OME-Zarr, ...)
  ChannelLayout             -- how channels are packed into a raw frame
  Aligner                   -- registration strategy (phase-corr, ECC, feature)
  Transform                 -- a fitted geometric mapping between channels
"""

from .movie import Movie, RawMovie, RGBMovie
from .frame import Frame, FrameBunch
from .channel import Channel, ScrubImage
from .layout import ChannelLayout, ChannelSpec
from .transform import Transform
from .registration import Aligner, PhaseCorrelationAligner
from .io_backends import MovieReader, MovieWriter, TiffReader, TiffWriter
from .pipeline import MergePipeline

__all__ = [
    "Movie", "RawMovie", "RGBMovie",
    "Frame", "FrameBunch",
    "Channel", "ScrubImage",
    "ChannelLayout", "ChannelSpec",
    "Transform",
    "Aligner", "PhaseCorrelationAligner",
    "MovieReader", "MovieWriter", "TiffReader", "TiffWriter",
    "MergePipeline",
]
