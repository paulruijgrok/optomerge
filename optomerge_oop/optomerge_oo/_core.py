"""
optomerge_oo._core
================
Bootstrap access to the existing functional ``optomerge`` package.

The OO layer is a thin, safe shell over the validated numerical routines in the
original package; it does not re-implement the maths. This module locates that
package (without requiring it to be pip-installed) and re-exports the functions
the OO classes call.

If you later `pip install` the original package, the ``sys.path`` insertion
below becomes a harmless no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ffreg/optomerge_oop/optomerge_oo/_core.py  ->  ffreg/optomerge  (package root)
_FUNCTIONAL_ROOT = Path(__file__).resolve().parents[2] / "optomerge"
if _FUNCTIONAL_ROOT.is_dir() and str(_FUNCTIONAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_FUNCTIONAL_ROOT))

# Re-export the functional core. Import submodules directly so we depend only on
# the functions we use.
from optomerge.io import (  # noqa: E402
    load_frames,
    get_movie_info,
    save_rgb_tiff,
    save_tiff,
)
from optomerge.processing import (  # noqa: E402
    norm_image,
    auto_norm_image,
    subtract_background,
    crop_channel,
)
from optomerge.registration import calculate_alignment  # noqa: E402
from optomerge.segmentation import find_channel_bounds  # noqa: E402
from optomerge.transform import transform_image, zero_pad_images  # noqa: E402

__all__ = [
    "load_frames", "get_movie_info", "save_rgb_tiff", "save_tiff",
    "norm_image", "auto_norm_image", "subtract_background", "crop_channel",
    "calculate_alignment", "find_channel_bounds",
    "transform_image", "zero_pad_images",
]
