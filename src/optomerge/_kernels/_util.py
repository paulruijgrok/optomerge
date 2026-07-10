"""Small shared helpers for the kernels."""
from __future__ import annotations

import numpy as np


def _as_float(a: np.ndarray) -> np.ndarray:
    """Return ``a`` as a floating array, *preserving* float32 vs float64.

    The kernels historically forced float64. Preserving the input floating dtype
    lets the merge run in float32 (halving its multi-GB intermediates) while
    integer / other inputs are still promoted to float64 for safe arithmetic.
    """
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating):
        return a
    return a.astype(np.float64)
