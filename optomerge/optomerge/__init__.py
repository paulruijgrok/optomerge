"""
optomerge
=========
Crop and register multichannel sub-images from raw fluorescence movie stacks.

Algorithmic equivalent of the MATLAB ``align`` toolbox (Ruijgrok / Nakamura).

Public API
----------
Classes
~~~~~~~
    AlignmentPipeline   – high-level pipeline: load → find channels →
                          calculate alignment → apply to all frames → save
    SharedAlignment     – holds channel bounds + alignment transform for
                          reuse across a batch of movies
    OptomergeConfig     – Pydantic root config; load from TOML or use defaults

Functions
~~~~~~~~~
    find_channel_bounds   – locate imaging-channel boundaries in a projection
    calculate_alignment   – FFT phase-correlation + scale/rotation search
    transform_image       – apply affine transform (rot, sx, sy, tx, ty)
    zero_pad_images       – zero-pad two images to next-power-of-2 size
    norm_image            – linear intensity normalisation to [0, 1]
    subtract_background   – morphological-opening background subtraction
    crop_channel          – crop to bounds and scrub outside pixels
    load_frames           – read a frame range from a TIFF movie
    save_tiff             – write a numpy array as a multi-page TIFF

"""

from .pipeline import AlignmentPipeline, SharedAlignment
from .config import OptomergeConfig
from .segmentation import find_channel_bounds
from .registration import calculate_alignment
from .transform import transform_image, zero_pad_images
from .processing import norm_image, subtract_background, crop_channel
from .io import load_frames, save_tiff

__all__ = [
    "AlignmentPipeline",
    "SharedAlignment",
    "OptomergeConfig",
    "find_channel_bounds",
    "calculate_alignment",
    "transform_image",
    "zero_pad_images",
    "norm_image",
    "subtract_background",
    "crop_channel",
    "load_frames",
    "save_tiff",
]
