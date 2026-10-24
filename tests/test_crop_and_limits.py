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


# --------------------------------------------------------------------------- #
# "objects" normalisation
# --------------------------------------------------------------------------- #

def _object_frames(n=6, amp=200.0, bright=None):
    """Noisy background (~100) with a few 3x3 objects of height ``amp``; optional bright blob."""
    rng = np.random.default_rng(2)
    f = 100.0 + rng.normal(0, 3, (80, 80, n))
    for k in range(n):
        for r, c in ((10 + 5 * k, 15), (40, 10 + 6 * k), (60, 60)):
            f[r:r + 3, c:c + 3, k] += amp
    if bright is not None:
        f[20:25, 50:55, :] += bright
    return f


def test_object_amplitude_finds_the_typical_object_height():
    pytest.importorskip("scipy")
    from optomerge.calibration import _object_amplitude
    from optomerge._core import subtract_background
    amp = _object_amplitude(subtract_background(_object_frames().astype(np.float32), radius=5))
    assert amp == pytest.approx(200.0, rel=0.1)


def test_objects_mode_puts_the_median_object_at_the_target_and_saturates_blobs():
    """End to end through the merge: typical objects at 0.5, a 20x blob just saturates.

    The blob has nearly as many pixels as all the objects together, so a
    per-pixel median would land on it; one vote per object must not.
    """
    pytest.importorskip("scipy")
    frames = _object_frames(bright=4000.0)
    limits = _compute_limits(_OneChannelLayout(), frames.max(axis=2), 0.0, 2.0,
                             norm_mode="objects", frames=frames, bg_radius=5, norm_target=0.5)
    ch = _OneChannelLayout().split(frames)[0]
    ch.vmin, ch.vmax = limits["green"]
    out = RGBMovie.from_channels([ch], {}, bg_radius=5).to_array()[:, :, 1, :]
    assert np.median(out[61, 61, :]) == pytest.approx(0.5, abs=0.06)   # an object centre
    assert out[22, 52, :].min() >= 1.0                                 # the blob saturates


def test_objects_mode_falls_back_to_range_for_an_empty_channel():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(3)
    frames = 100.0 + rng.normal(0, 3, (80, 80, 6))
    notes = []
    vmin, vmax = _compute_limits(_OneChannelLayout(), frames.max(axis=2), 0.0, 2.0, notes,
                                 norm_mode="objects", frames=frames, bg_radius=5)["green"]
    assert "no objects detected" in notes[0]
    assert vmax <= frames.max() + 1e-9


def test_objects_on_a_raised_background_still_reach_the_target():
    """Uneven illumination: objects on a bright background patch must not be clipped flat.

    The merge normalises before background subtraction; with an "objects" vmax
    (well below background + object) an upper clip there would flatten them.
    """
    pytest.importorskip("scipy")
    frames = _object_frames()
    frames[30:80, 30:80, :] += 800.0                     # bright half of the field
    limits = _compute_limits(_OneChannelLayout(), frames.max(axis=2), 0.0, 2.0,
                             norm_mode="objects", frames=frames, bg_radius=5, norm_target=0.5)
    ch = _OneChannelLayout().split(frames)[0]
    ch.vmin, ch.vmax = limits["green"]
    out = RGBMovie.from_channels([ch], {}, bg_radius=5).to_array()[:, :, 1, :]
    assert np.median(out[61, 61, :]) == pytest.approx(0.5, abs=0.06)   # object on the patch


def test_objects_mode_keeps_noisy_background_out_of_the_lower_clip():
    """vmin must sit below the per-frame background, not at the max projection's minimum.

    With real noise, the max projection's minimum is ~3 sigma above the typical
    background (MyLOV1Char: 87-94 % of raw pixels below it). Clipping there
    measures objects from that level instead of from their local background.
    """
    pytest.importorskip("scipy")
    rng = np.random.default_rng(4)
    frames = 1000.0 + rng.normal(0, 20, (80, 80, 40))
    for k in range(40):
        for r, c in ((10, 10 + k), (40 + k // 2, 30), (60, 60)):
            frames[r:r + 3, c:c + 3, k] += 300.0
    limits = _compute_limits(_OneChannelLayout(), frames.max(axis=2), 0.0, 2.0,
                             norm_mode="objects", frames=frames, bg_radius=5, norm_target=0.5)
    vmin, _ = limits["green"]
    assert vmin < np.median(frames)                       # background survives the clip
    ch = _OneChannelLayout().split(frames)[0]
    ch.vmin, ch.vmax = limits["green"]
    out = RGBMovie.from_channels([ch], {}, bg_radius=5).to_array()[:, :, 1, :]
    assert np.median(out[61, 61, :]) == pytest.approx(0.5, abs=0.08)
