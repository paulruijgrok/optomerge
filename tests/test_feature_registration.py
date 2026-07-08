"""Tests for the FeatureDistanceAligner (head-to-filament registration).

The objective/grid-search/sign-convention logic is exercised with a synthetic
distance map and a monkeypatched detector, so it runs without scipy. The full
detection path (threshold -> connected components -> centroids -> distance
transform) needs scipy and is covered by ``test_end_to_end_recovers_shift``,
which skips cleanly when scipy is absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from optomerge import FeatureDistanceAligner, PhaseCorrelationAligner, Settings
from optomerge.registration import Aligner


# --------------------------------------------------------------------------- #
# Objective + grid search + sign convention (no scipy needed)
# --------------------------------------------------------------------------- #

def _dot_dt(shape, r0, c0):
    """Euclidean distance to a single filament dot at (r0, c0)."""
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    return np.sqrt((rr - r0) ** 2 + (cc - c0) ** 2)


def test_recovers_diagonal_shift(monkeypatch):
    # Filament dot at (50, 50); heads at (47, 47). transform_image moves a head by
    # (-t2, -t1), so landing (47,47) on (50,50) needs t = head - filament = (-3, -3).
    dt = _dot_dt((100, 100), 50, 50)
    per_frame = [(np.array([47.0]), np.array([47.0]), dt) for _ in range(4)]
    a = FeatureDistanceAligner(max_shift=10, step=1.0)
    monkeypatch.setattr(a, "_detect", lambda ref, mov: per_frame)
    t = a.align(object(), object())
    assert abs(t.t1 - (-3.0)) < 0.6   # t1 -> columns
    assert abs(t.t2 - (-3.0)) < 0.6   # t2 -> rows
    assert t.score > 0.9              # mean distance ~0 at the optimum


def test_recovers_single_axis_shift(monkeypatch):
    # Head at col 45, filament at col 50: t1 = head - filament = -5, t2 = 0.
    dt = _dot_dt((100, 100), 50, 50)
    per_frame = [(np.array([50.0]), np.array([45.0]), dt) for _ in range(3)]
    a = FeatureDistanceAligner(max_shift=12, step=1.0)
    monkeypatch.setattr(a, "_detect", lambda ref, mov: per_frame)
    t = a.align(object(), object())
    assert abs(t.t1 - (-5.0)) < 0.6
    assert abs(t.t2 - 0.0) < 0.6


def test_robust_to_orphan_heads(monkeypatch):
    # One filament dot at (50, 50). Real heads at (47, 47) need shift (+3, +3);
    # orphan heads sit at (10, 90), far from any filament at any small shift.
    dt = _dot_dt((100, 100), 50, 50)
    rows = np.array([47.0, 47.0, 10.0, 10.0])   # 2 inliers + 2 orphans
    cols = np.array([47.0, 47.0, 90.0, 90.0])
    per_frame = [(rows, cols, dt) for _ in range(4)]
    a = FeatureDistanceAligner(max_shift=10, step=1.0, distance_cap=6.0)
    monkeypatch.setattr(a, "_detect", lambda ref, mov: per_frame)
    t = a.align(object(), object())
    # Orphans (capped) must not drag the fit away from the real heads.
    # Inliers (47,47) onto (50,50): t = head - filament = (-3, -3).
    assert abs(t.t1 - (-3.0)) < 0.6 and abs(t.t2 - (-3.0)) < 0.6
    # Score reflects the inlier fraction (~half the heads are on filament).
    assert 0.3 < t.score < 0.7


def test_stuck_weights_downweight():
    # A head stuck at (5,5) across 3 frames should weight 1/3; unique heads weight 1.
    dummy = np.zeros((2, 2))
    per_frame = [
        (np.array([5.0, 10.0]), np.array([5.0, 40.0]), dummy),
        (np.array([5.0, 20.0]), np.array([5.0, 40.0]), dummy),
        (np.array([5.0, 30.0]), np.array([5.0, 40.0]), dummy),
    ]
    a = FeatureDistanceAligner(max_shift=8, deweight_stuck=True, stuck_radius=2.0)
    w = a._stuck_weights(per_frame)
    np.testing.assert_allclose(w, [1/3, 1, 1/3, 1, 1/3, 1], atol=1e-9)
    # Disabled -> all ones.
    a2 = FeatureDistanceAligner(max_shift=8, deweight_stuck=False)
    np.testing.assert_allclose(a2._stuck_weights(per_frame), np.ones(6))


def test_deweight_stuck_overrides_dominant_stuck_object(monkeypatch):
    # Filament = vertical line at col 50 -> distance depends only on column.
    H = W = 60
    dt = np.abs(np.arange(W)[None, :] - 50).repeat(H, axis=0).astype(float)
    # Stuck head at (5,55) in all 10 frames -> wants t1 = col-50 = +5 (lands col 50).
    # Three moving inlier heads at col 47 (unique rows) -> want t1 = -3.
    per_frame = []
    for k in range(10):
        rows = [5.0]; cols = [55.0]
        if k < 3:
            rows.append(10.0 + 10 * k); cols.append(47.0)
        per_frame.append((np.array(rows), np.array(cols), dt))

    a_off = FeatureDistanceAligner(max_shift=8, step=1.0, distance_cap=50,
                                   deweight_stuck=False)
    a_off._detect = lambda ref, mov: per_frame
    assert a_off.align(object(), object()).t1 > 3        # stuck object dominates -> ~+5

    a_on = FeatureDistanceAligner(max_shift=8, step=1.0, distance_cap=50,
                                  deweight_stuck=True, stuck_radius=2.0)
    a_on._detect = lambda ref, mov: per_frame
    assert a_on.align(object(), object()).t1 < -1        # moving heads win -> ~-3


def test_no_features_returns_identity(monkeypatch):
    a = FeatureDistanceAligner(max_shift=8)
    monkeypatch.setattr(a, "_detect", lambda ref, mov: [])
    t = a.align(object(), object())
    assert t.t1 == 0.0 and t.t2 == 0.0 and t.rot == 0.0


def test_translation_only_output(monkeypatch):
    dt = _dot_dt((60, 60), 30, 30)
    per_frame = [(np.array([28.0]), np.array([31.0]), dt)]
    a = FeatureDistanceAligner(max_shift=6, step=1.0)
    monkeypatch.setattr(a, "_detect", lambda ref, mov: per_frame)
    t = a.align(object(), object())
    assert t.rot == 0.0 and t.s1 == 1.0 and t.s2 == 1.0


def test_requires_positive_max_shift():
    with pytest.raises(ValueError):
        FeatureDistanceAligner(max_shift=0)


# --------------------------------------------------------------------------- #
# Config / seam wiring
# --------------------------------------------------------------------------- #

def test_needs_frames_flag():
    assert Aligner.needs_frames is False
    assert PhaseCorrelationAligner().needs_frames is False
    assert FeatureDistanceAligner(max_shift=30).needs_frames is True


def test_config_selects_feature_aligner():
    s = Settings.from_sources({"alignment": {
        "method": "feature", "head_sigma": 4.0, "max_shift": 20,
        "head_min_area": 5, "head_max_area": 80, "filament_min_area": 30,
        "deweight_stuck": True, "stuck_radius": 3.0,
    }})
    a = s.build_aligner()
    assert isinstance(a, FeatureDistanceAligner)
    assert a.max_shift == 20 and a.head_sigma == 4.0
    assert a.min_head_area == 5 and a.max_head_area == 80 and a.filament_min_area == 30
    assert a.deweight_stuck is True and a.stuck_radius == 3.0
    # Default calibrator threads feature_frames through (default bumped to 60).
    assert s.build_calibrator().feature_frames == 60


def test_config_default_is_phase():
    assert isinstance(Settings().build_aligner(), PhaseCorrelationAligner)


def test_feature_unconstrained_max_shift_defaults_to_30():
    s = Settings.from_sources({"alignment": {"method": "feature"}})  # max_shift left 0
    assert s.build_aligner().max_shift == 30.0


# --------------------------------------------------------------------------- #
# Full detection path (needs scipy)
# --------------------------------------------------------------------------- #

def test_detection_filters_noise():
    """Filament despeckle + head area gate reject speckle and oversized blobs."""
    pytest.importorskip("scipy")
    H = W = 50
    green = np.zeros((H, W))
    green[24:27, 5:45] = 5.0            # a real filament bar (large component)
    green[2, 2] = green[40, 40] = 9.0   # two isolated bright specks -> despeckled out
    red = np.zeros((H, W))
    red[25, 20:22] = 8.0                # a genuine small head (~2-6 px)
    red[10, 10] = 8.0                   # a 1-px noise speck -> below min_head_area
    red[5:15, 30:40] = 8.0              # a big 100-px blob -> above max_head_area

    a = FeatureDistanceAligner(max_shift=6, min_head_area=2, max_head_area=60,
                               filament_min_area=20)
    rows, cols, fil = a.frame_features(green, red)
    # Filament mask keeps the bar but not the isolated specks.
    assert fil[25, 25] and not fil[2, 2] and not fil[40, 40]
    # Exactly the genuine head survives the area gate (noise + big blob rejected).
    assert rows.size == 1
    assert abs(rows[0] - 25) <= 1 and 19 <= cols[0] <= 22


def test_end_to_end_recovers_shift():
    """Fit, then actually apply the transform: the head must land on the filament.

    This is the regression guard for the transform sign -- it fails if the fitted
    translation is applied in the wrong direction.
    """
    pytest.importorskip("scipy")
    from optomerge.channel import Channel
    from optomerge._kernels.transform import transform_image

    H = W = 40
    n = 5
    bounds = np.array([[0, H - 1], [0, W - 1]])
    # Filament: a bright horizontal bar at rows 19-21; head displaced +4 rows off it.
    green = np.zeros((H, W, n))
    green[19:22, 5:35, :] = 6.0
    off_r = 4
    head_r, head_c = 20 + off_r, 20
    red = np.zeros((H, W, n))
    red[head_r - 1:head_r + 2, head_c - 1:head_c + 2, :] = 9.0  # 3x3 blob (area 9)

    ref = Channel(green, "green", bounds, color="green", reference=True)
    mov = Channel(red, "red", bounds, color="red")
    a = FeatureDistanceAligner(max_shift=8, step=1.0)
    t = a.align(ref, mov)

    # Apply the fitted transform to the head frame; it must move onto the bar.
    warped = transform_image(red[:, :, 0], 0.0, 1.0, 1.0, tx=t.t1, ty=t.t2,
                             backend="scipy", n_workers=1)
    r_land, c_land = np.unravel_index(int(np.argmax(warped)), warped.shape)
    assert 19 <= r_land <= 21, f"head landed at row {r_land}, not on the bar (19-21)"
    assert 5 <= c_land <= 34
    assert np.isfinite(t.score)
