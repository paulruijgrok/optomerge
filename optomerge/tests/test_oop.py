"""
Smoke tests for the object-oriented optomerge public API.

The pure-Python tests run anywhere. The end-to-end pipeline test needs the full
numerical stack (scipy, scikit-image, tifffile) and a movie under ``test_data/``;
it is skipped automatically when either is unavailable.

Run with:  pytest tests/
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from optomerge import (
    ChannelLayout,
    FrameBunch,
    MergePipeline,
    PhaseCorrelationAligner,
    Transform,
)
from optomerge._kernels.transform import _next_pow2, zero_pad_images


# --------------------------------------------------------------------------- #
# Pure-Python API (no heavy deps)
# --------------------------------------------------------------------------- #

class TestTransform:
    def test_identity_is_noop(self):
        t = Transform()
        assert t.is_identity
        img = np.arange(16.0).reshape(4, 4)
        np.testing.assert_array_equal(t.apply(img), img)

    def test_matrix_shape_and_translation(self):
        m = Transform(t1=3.0, t2=-2.0).matrix()
        assert m.shape == (3, 3)
        assert m[0, 2] == 3.0 and m[1, 2] == -2.0

    def test_rescaled_scales_translation_only(self):
        t = Transform(t1=4.0, t2=8.0, rot=0.1, s1=1.2, s2=0.9)
        r = t.rescaled(0.5)
        assert (r.t1, r.t2) == (2.0, 4.0)
        assert (r.rot, r.s1, r.s2) == (t.rot, t.s1, t.s2)


class TestLayout:
    def test_factories_map_to_channel_orders(self):
        assert ChannelLayout.auto().channel_order == "auto"
        assert ChannelLayout.top_green_bottom_red().channel_order == "top_green_fils_bottom_red_heads"
        assert ChannelLayout.top_red_bottom_green().channel_order == "top_red_heads_bottom_green_fils"

    def test_unresolved_layout_cannot_split(self):
        with pytest.raises(RuntimeError):
            ChannelLayout.auto().split(np.zeros((10, 10)))


class TestFrameBunch:
    def test_promotes_2d_to_3d(self):
        fb = FrameBunch(np.zeros((8, 8)), start_index=0)
        assert fb.data.ndim == 3 and len(fb) == 1

    def test_projections(self):
        data = np.stack([np.full((4, 4), i) for i in range(5)], axis=2)
        fb = FrameBunch(data, start_index=10)
        assert fb.max_projection().max() == 4
        np.testing.assert_allclose(fb.mean_projection(), np.full((4, 4), 2.0))


class TestKernelHelpers:
    def test_next_pow2(self):
        assert _next_pow2(100) == 128
        assert _next_pow2(128) == 256  # original rule: pads even exact powers

    def test_zero_pad_to_common_power_of_2(self):
        a, b = zero_pad_images(np.ones((50, 60)), np.ones((40, 55)))
        assert a.shape == b.shape
        H, W = a.shape
        assert H == _next_pow2(50) and W == _next_pow2(60)


# --------------------------------------------------------------------------- #
# End-to-end pipeline (needs scipy/skimage/tifffile + a test movie)
# --------------------------------------------------------------------------- #

def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _first_test_movie() -> Path | None:
    data_dir = Path(__file__).resolve().parents[2] / "test_data"
    if not data_dir.is_dir():
        return None
    tifs = sorted(data_dir.rglob("*.tif")) + sorted(data_dir.rglob("*.tiff"))
    return tifs[0] if tifs else None


@pytest.mark.skipif(
    not (_have("scipy") and _have("skimage") and _have("tifffile")),
    reason="requires scipy, scikit-image and tifffile",
)
def test_pipeline_end_to_end():
    movie = _first_test_movie()
    if movie is None:
        pytest.skip("no test movie under test_data/")
    pipe = MergePipeline(movie, aligner=PhaseCorrelationAligner(), projection_frames=50)
    rgb = pipe.run().to_array()
    assert rgb.ndim == 4 and rgb.shape[2] == 3       # (H, W, 3, N)
    assert rgb.shape[3] > 0                           # at least one frame
    assert pipe.resolved_layout is not None
