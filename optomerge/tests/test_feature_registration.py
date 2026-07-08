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
    # Filament dot at (50, 50); heads recorded at (47, 47) -> need +3 row, +3 col.
    dt = _dot_dt((100, 100), 50, 50)
    per_frame = [(np.array([47.0]), np.array([47.0]), dt) for _ in range(4)]
    a = FeatureDistanceAligner(max_shift=10, step=1.0)
    monkeypatch.setattr(a, "_detect", lambda ref, mov: per_frame)
    t = a.align(object(), object())
    assert abs(t.t1 - 3.0) < 0.6   # t1 -> columns
    assert abs(t.t2 - 3.0) < 0.6   # t2 -> rows
    assert t.score > 0.9           # mean distance ~0 at the optimum


def test_recovers_single_axis_shift(monkeypatch):
    # Heads offset in columns only (45 -> 50): expect t1=+5, t2=0.
    dt = _dot_dt((100, 100), 50, 50)
    per_frame = [(np.array([50.0]), np.array([45.0]), dt) for _ in range(3)]
    a = FeatureDistanceAligner(max_shift=12, step=1.0)
    monkeypatch.setattr(a, "_detect", lambda ref, mov: per_frame)
    t = a.align(object(), object())
    assert abs(t.t1 - 5.0) < 0.6
    assert abs(t.t2 - 0.0) < 0.6


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
    }})
    a = s.build_aligner()
    assert isinstance(a, FeatureDistanceAligner)
    assert a.max_shift == 20 and a.head_sigma == 4.0
    assert a.min_head_area == 5 and a.max_head_area == 80 and a.filament_min_area == 30
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
    pytest.importorskip("scipy")
    from optomerge.channel import Channel

    H = W = 40
    n = 5
    bounds = np.array([[0, H - 1], [0, W - 1]])
    # Filament: a bright horizontal bar; head: a bright dot sitting on it, but the
    # recorded red dot is offset by (+4 row, -3 col) from the true filament point.
    green = np.zeros((H, W, n))
    green[19:22, 5:35, :] = 6.0
    true_r, true_c = 20, 20
    off_r, off_c = 4, -3           # red head displaced from its true position
    red = np.zeros((H, W, n))
    red[true_r + off_r, true_c + off_c, :] = 9.0

    ref = Channel(green, "green", bounds, color="green", reference=True)
    mov = Channel(red, "red", bounds, color="red")
    a = FeatureDistanceAligner(max_shift=8, step=1.0)
    t = a.align(ref, mov)
    # To land the head back on the bar we must shift it by (-off_r, -off_c).
    assert abs(t.t2 - (-off_r)) <= 1.0    # rows
    # Column: the bar spans many columns, so any small |t1| that keeps the head
    # on the bar is acceptable; just check it stays bounded and finite.
    assert abs(t.t1) <= a.max_shift
    assert np.isfinite(t.score)
