"""Output crop rule (footprint vs signal) and the outlier cap on normalisation limits."""
from __future__ import annotations

import numpy as np
import pytest

from optomerge.calibration import _compute_limits
from optomerge.channel import Channel
from optomerge.movie import RGBMovie
from optomerge.transform import Transform


# --------------------------------------------------------------------------- #
# vmax outlier cap
# --------------------------------------------------------------------------- #

class _OneChannelLayout:
    """Minimal resolved layout: the whole image is one channel."""

    def split(self, image):
        h, w = image.shape[:2]
        return [Channel(image, "green", np.array([[0, h - 1], [0, w - 1]]),
                        color="green", reference=True)]


def _projection(outlier: bool) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = 100.0 + rng.normal(0, 2, (200, 200))   # 40k px: p99.9 = top 40 px
    img[10:30, 10:30] = 400.0                    # real signal: 400 px (1% of field)
    if outlier:
        img[100:105, 100:105] = 4000.0           # one bright aggregate (25 px < 0.1%)
    return img


def test_cap_tames_a_bright_outlier_and_records_a_note():
    pytest.importorskip("scipy")
    notes = []
    vmin, vmax = _compute_limits(_OneChannelLayout(), _projection(True), 0.0, 2.0, notes)["green"]
    # p99.9 sits on the real signal (400), so the cap is vmin + 2 * (400 - vmin).
    assert vmax == pytest.approx(vmin + 2 * (400.0 - vmin), rel=1e-6)
    assert len(notes) == 1 and "green: vmax capped" in notes[0]


def test_cap_leaves_normal_channels_and_can_be_disabled():
    pytest.importorskip("scipy")
    notes = []
    _, vmax = _compute_limits(_OneChannelLayout(), _projection(False), 0.0, 2.0, notes)["green"]
    assert vmax == pytest.approx(400.0) and notes == []
    _, vmax_off = _compute_limits(_OneChannelLayout(), _projection(True), 0.0, 0.0, notes)["green"]
    assert vmax_off == pytest.approx(4000.0) and notes == []


# --------------------------------------------------------------------------- #
# output crop
# --------------------------------------------------------------------------- #

def _sparse_pair(h=40, w=60, n=3):
    """Two channel crops with one small object near a corner and nothing else."""
    bounds = np.array([[0, h - 1], [0, w - 1]])
    g = np.zeros((h, w, n))
    g[5:9, 5:9, :] = 1.0
    r = g.copy()
    chans = []
    for data, name, ref in ((g, "green", True), (r, "red", False)):
        c = Channel(data, name, bounds, color=name, reference=ref)
        c.vmin, c.vmax = 0.0, 1.0
        chans.append(c)
    return chans


def test_footprint_crop_keeps_the_whole_field_of_a_sparse_movie():
    chans = _sparse_pair()
    footprint = RGBMovie.from_channels(chans, {"red": Transform()}, bg_radius=2)
    signal = RGBMovie.from_channels(_sparse_pair(), {"red": Transform()}, bg_radius=2,
                                    crop="signal")
    assert footprint.to_array().shape[:2] == (40, 60)
    h, w = signal.to_array().shape[:2]
    assert h < 15 and w < 15                     # legacy rule hugs the one object


def test_footprint_crop_trims_what_the_warp_leaves_uncovered():
    # Shifting red by 3 px leaves a 3-px column without red data; the crop drops it.
    out = RGBMovie.from_channels(_sparse_pair(), {"red": Transform(t1=3.0)}, bg_radius=2)
    assert out.to_array().shape[:2] == (40, 57)


def test_unknown_crop_rule_is_rejected():
    with pytest.raises(ValueError):
        RGBMovie.from_channels(_sparse_pair(), {"red": Transform()}, crop="tight")
