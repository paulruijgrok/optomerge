"""Tests for the row-profile channel segmentation (pure numpy).

The row-profile path does not call scipy/k-means, so these exercise real
behaviour on synthetic two-band images.
"""
from __future__ import annotations

import numpy as np

from optomerge._kernels.segmentation import (
    _band_extent,
    _find_bounds_row_profile,
    _find_split,
    _rect_scrub,
    _smooth1d,
    find_channel_bounds,
)


def _two_band_image(H=200, W=240, top=(30, 80), bot=(120, 170),
                    cols=(20, 220), top_val=1.0, bot_val=0.5):
    im = np.zeros((H, W))
    im[top[0]:top[1], cols[0]:cols[1]] = top_val
    im[bot[0]:bot[1], cols[0]:cols[1]] = bot_val
    return im


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def test_band_extent_trims_dark_margins():
    prof = np.zeros(100)
    prof[30:70] = 1.0
    lo, hi = _band_extent(prof, 0, 99, trim_frac=0.2)
    assert lo == 30 and hi == 69


def test_find_split_locates_gap():
    prof = np.zeros(200)
    prof[30:80] = 1.0
    prof[120:170] = 0.6           # gap between 80 and 120
    split, _ = _find_split(prof, 200, search_frac=0.5)
    assert 80 <= split <= 120


def test_rect_scrub_masks_outside():
    s = _rect_scrub(10, 20, 5, 15, W=30, H=30)
    assert s.shape == (30, 30)
    assert not s[10:21, 5:16].any()      # inside kept
    assert s[0, 0] and s[29, 29]         # outside scrubbed


def test_smooth1d_preserves_length():
    x = np.random.default_rng(0).random(101)
    assert _smooth1d(x, 5).shape == x.shape


# --------------------------------------------------------------------------- #
# Row-profile segmentation
# --------------------------------------------------------------------------- #

def test_two_bands_detected_axis_aligned():
    im = _two_band_image()
    b1, s1, b2, s2 = find_channel_bounds(im, mean_projection=im)   # default = row_profile
    assert b2 is not None and s2 is not None
    assert s1.shape == im.shape
    # green = brighter = top band
    assert 28 <= b1[0, 0] <= 32 and 76 <= b1[0, 1] <= 82
    assert 118 <= b2[0, 0] <= 122 and 166 <= b2[0, 1] <= 172
    # columns trimmed to the illuminated width
    assert 18 <= b1[1, 0] <= 22 and 216 <= b1[1, 1] <= 221


def test_explicit_order_flips_assignment():
    im = _two_band_image()
    # Force red-on-top: channel 1 (green) should become the bottom band.
    b1, s1, b2, s2 = find_channel_bounds(
        im, mean_projection=im, channel_order="top_red_heads_bottom_green_fils")
    assert b1[0, 0] >= 118    # green now bottom
    assert b2[0, 0] <= 82     # red now top


def test_single_band_one_channel():
    im = np.zeros((200, 200))
    im[40:160, 20:180] = 1.0          # one uniform band, no gap
    b1, s1, b2, s2 = find_channel_bounds(im, mean_projection=im)
    assert b2 is None and s2 is None
    assert 38 <= b1[0, 0] <= 42 and 158 <= b1[0, 1] <= 162


def test_off_center_split():
    # Unequal bands: gap around row 70 rather than the H/2 midline.
    im = _two_band_image(H=200, top=(20, 65), bot=(80, 180), top_val=1.0, bot_val=0.8)
    b1, s1, b2, s2 = find_channel_bounds(im, mean_projection=im)
    assert b2 is not None
    assert b1[0, 1] <= 70 and b2[0, 0] >= 70    # split respected, not forced to 100


def test_column_trim_narrow_channel():
    im = _two_band_image(cols=(60, 160))
    b1, _, _, _ = find_channel_bounds(im, mean_projection=im)
    assert 58 <= b1[1, 0] <= 62 and 156 <= b1[1, 1] <= 161
