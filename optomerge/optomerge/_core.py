"""
optomerge._core
===============
Internal access point to the numerical core.

The object-oriented layer is a thin, typed shell over the validated, function
based image-processing routines that live in :mod:`optomerge._kernels`. This
module re-exports the handful of functions the OO classes call, so the rest of
the package depends on this single seam rather than on individual kernels.
"""

from __future__ import annotations

# Re-export the numerical core. Import submodules directly so we depend only on
# the functions we actually use.
from ._kernels.io import (
    load_frames,
    get_movie_info,
    save_rgb_tiff,
    save_tiff,
)
from ._kernels.processing import (
    norm_image,
    auto_norm_image,
    subtract_background,
    crop_channel,
)
from ._kernels.registration import calculate_alignment
from ._kernels.segmentation import find_channel_bounds
from ._kernels.transform import transform_image, zero_pad_images, _next_pow2

__all__ = [
    "load_frames", "get_movie_info", "save_rgb_tiff", "save_tiff",
    "norm_image", "auto_norm_image", "subtract_background", "crop_channel",
    "calculate_alignment", "find_channel_bounds",
    "transform_image", "zero_pad_images", "_next_pow2",
]
