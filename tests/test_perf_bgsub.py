"""Tests for the windowed background-subtraction optimisation (Phase 1).

`subtract_background_windowed` restricts the (spatially local) background
subtraction to the non-zero content window instead of running it over the whole
zero-padded merge canvas. For the default `morph_open` method the result must be
bit-identical to the full `subtract_background`; the merge output must therefore
be unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from optomerge._kernels.processing import (
    subtract_background,
    subtract_background_windowed,
)


def _padded_movie(seed=0):
    """A large mostly-zero canvas with a small block of content (a padded frame)."""
    rng = np.random.default_rng(seed)
    mov = np.zeros((256, 384, 6))
    mov[70:150, 90:230, :] = rng.random((80, 140, 6)) + 0.2  # off-centre content
    return mov


@pytest.mark.parametrize("backend", ["scipy", "cv2"])
def test_windowed_morph_open_bit_identical(backend):
    if backend == "cv2":
        pytest.importorskip("cv2")
    mov = _padded_movie()
    full = subtract_background(mov, radius=10, method="morph_open",
                              backend=backend, n_workers=1)
    win = subtract_background_windowed(mov, radius=10, method="morph_open",
                                       backend=backend, n_workers=1)
    # Grey opening depends only on inputs within 2*radius; the window is expanded
    # by exactly that, so the result must match to the bit.
    np.testing.assert_allclose(win, full, atol=0.0, rtol=0.0)


def test_windowed_gaussian_close():
    mov = _padded_movie()
    full = subtract_background(mov, radius=8, method="gaussian",
                              backend="scipy", n_workers=1)
    win = subtract_background_windowed(mov, radius=8, method="gaussian",
                                       backend="scipy", n_workers=1)
    # Gaussian has infinite support; the 4-sigma border makes the tail negligible.
    np.testing.assert_allclose(win, full, atol=1e-6)


def test_windowed_all_zero_movie():
    mov = np.zeros((64, 64, 3))
    win = subtract_background_windowed(mov, radius=10)
    assert np.array_equal(win, np.zeros_like(mov))


def test_windowed_single_frame_2d():
    mov = _padded_movie()[:, :, 0]
    full = subtract_background(mov, radius=10, backend="scipy", n_workers=1)
    win = subtract_background_windowed(mov, radius=10, backend="scipy", n_workers=1)
    assert win.shape == mov.shape
    np.testing.assert_allclose(win, full, atol=0.0, rtol=0.0)


def test_merge_output_unchanged_by_windowing(monkeypatch):
    """RGBMovie.from_channels output must be identical with windowed vs full bg-sub.

    Guards that swapping in the windowed bg-sub did not change the merged movie:
    run the real path, then force the *full* subtract_background back in and
    compare the two RGB outputs.
    """
    from optomerge import movie as movie_mod
    from optomerge.channel import Channel
    from optomerge.transform import Transform

    H = W = 96
    n = 4
    rng = np.random.default_rng(1)
    bounds = np.array([[0, H - 1], [0, W - 1]])

    green = np.zeros((H, W, n))
    green[20:70, 15:80, :] = rng.random((50, 65, n)) + 0.3
    red = np.zeros((H, W, n))
    red[25:65, 20:75, :] = rng.random((40, 55, n)) + 0.3

    def make_channels():
        g = Channel(green.copy(), "green", bounds, color="green", reference=True)
        r = Channel(red.copy(), "red", bounds, color="red")
        g.vmin, g.vmax = 0.0, float(green.max())
        r.vmin, r.vmax = 0.0, float(red.max())
        return [g, r]

    transforms = {"red": Transform(t1=3.0, t2=-2.0, rot=np.deg2rad(1.0), s1=1.0, s2=1.0)}

    rgb_windowed = movie_mod.RGBMovie.from_channels(make_channels(), transforms,
                                                    bg_radius=10).to_array()

    # Force the pre-optimisation path (full bg-sub over the padded canvas).
    monkeypatch.setattr(movie_mod, "subtract_background_windowed",
                        movie_mod.subtract_background)
    rgb_full = movie_mod.RGBMovie.from_channels(make_channels(), transforms,
                                                bg_radius=10).to_array()

    assert rgb_windowed.shape == rgb_full.shape
    np.testing.assert_allclose(rgb_windowed, rgb_full, atol=0.0, rtol=0.0)
