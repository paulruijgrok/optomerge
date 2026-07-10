"""Tests for the float32 merge precision (Phase 3, memory footprint).

The merge runs in float32 by default to halve its multi-GB intermediates + RGB
buffer. This checks the kernels preserve the input floating dtype (so float32
flows end to end, while existing float64 callers are unchanged) and that the
merge honours ``dtype`` with a negligible effect on the 8-bit output.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from optomerge import RGBMovie
from optomerge._core import subtract_background, transform_image, zero_pad_images
from optomerge.channel import Channel
from optomerge.transform import Transform


@pytest.mark.parametrize("backend", ["scipy", "cv2"])
def test_kernels_preserve_float32(backend):
    if backend == "cv2":
        pytest.importorskip("cv2")
    a32 = (np.random.default_rng(0).random((40, 50, 6))).astype(np.float32)
    assert subtract_background(a32, radius=5, backend=backend, n_workers=1).dtype == np.float32
    assert transform_image(a32, 0.01, 1.0, 1.0, 2.0, -1.0,
                           backend=backend, n_workers=1).dtype == np.float32


def test_kernels_still_float64_for_float64_input():
    """Existing float64 callers are unchanged (dtype-preserving, not forced)."""
    a64 = np.random.default_rng(0).random((32, 40, 4))
    assert subtract_background(a64, radius=4, n_workers=1).dtype == np.float64
    assert transform_image(a64, 0.0, 1.0, 1.0, 1.0, 0.0, n_workers=1).dtype == np.float64
    p1, p2 = zero_pad_images(a64[:, :, 0], a64[:, :, 0])
    assert p1.dtype == np.float64


def test_zero_pad_preserves_float32():
    a = np.ones((20, 24), np.float32)
    p1, p2 = zero_pad_images(a, a)
    assert p1.dtype == np.float32 and p2.dtype == np.float32


def _channels():
    H = W = 96
    n = 4
    rng = np.random.default_rng(1)
    b = np.array([[0, H - 1], [0, W - 1]])
    g = np.zeros((H, W, n)); g[20:70, 15:80, :] = rng.random((50, 65, n)) + 0.3
    r = np.zeros((H, W, n)); r[25:65, 20:75, :] = rng.random((40, 55, n)) + 0.3
    gg = Channel(g, "green", b, color="green", reference=True)
    rr = Channel(r, "red", b, color="red")
    gg.vmin, gg.vmax = 0.0, float(g.max())
    rr.vmin, rr.vmax = 0.0, float(r.max())
    return [gg, rr]


def test_merge_defaults_to_float32_and_double_option():
    tf = {"red": Transform(t1=3.0, t2=-2.0, rot=np.deg2rad(1.0), s1=1.0, s2=1.0)}
    a32 = RGBMovie.from_channels(_channels(), tf).to_array()
    a64 = RGBMovie.from_channels(_channels(), tf, dtype=np.float64).to_array()
    assert a32.dtype == np.float32 and a64.dtype == np.float64
    # 8-bit render is what ships; it must be identical between the two precisions.
    s32 = (np.clip(a32, 0, 1) * 255).astype(np.uint8)
    s64 = (np.clip(a64, 0, 1) * 255).astype(np.uint8)
    np.testing.assert_array_equal(s32, s64)
