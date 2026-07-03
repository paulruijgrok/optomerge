"""Tests for the Calibrator strategies.

These use lightweight fake layout/aligner doubles so the best-of-N selection
logic is exercised without real segmentation (scipy/skimage). The end-to-end
path on a real movie is covered by test_oop.test_pipeline_end_to_end.
"""
from __future__ import annotations

import numpy as np
import pytest

from optomerge import (
    AcceptanceCriteria,
    AlignmentNotFoundError,
    RawMovie,
    RobustCalibrator,
    SingleProjectionCalibrator,
    Transform,
)

IMG = 512


# --------------------------------------------------------------------------- #
# Fakes: a layout that "resolves" to fixed specs, an aligner with scripted scores
# --------------------------------------------------------------------------- #

class _Spec:
    def __init__(self, name, reference, bounds):
        self.name = name
        self.color = name
        self.reference = reference
        self.bounds = bounds
        self.mask = None


class _Chan:
    def __init__(self, name, reference):
        self.name = name
        self.reference = reference
        self.vmin = None
        self.vmax = None

    def calibrate(self):
        self.vmin, self.vmax = 0.0, 1.0
        return self


class _Resolved:
    def __init__(self, specs):
        self.name = "fake"
        self.specs = specs

    @property
    def moving_specs(self):
        return [s for s in self.specs if not s.reference]

    def split(self, image):
        return [_Chan(s.name, s.reference) for s in self.specs]


class _Layout:
    def __init__(self, specs):
        self._specs = specs

    def resolve(self, max_proj, mean_proj=None, verbose=False):
        return _Resolved(self._specs)


class _ScriptedAligner:
    """Returns transforms with a scripted sequence of cross-correlation scores."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.i = 0

    def align(self, reference, moving):
        s = self.scores[min(self.i, len(self.scores) - 1)]
        self.i += 1
        return Transform(t1=1.0, t2=1.0, rot=0.0, s1=1.0, s2=1.0, score=s)


def _good_specs():
    return [
        _Spec("green", True, np.array([[10, 245], [20, 480]])),
        _Spec("red", False, np.array([[266, 500], [20, 480]])),
    ]


def _tiny_specs():
    return [
        _Spec("green", True, np.array([[10, 40], [20, 60]])),   # far too small
        _Spec("red", False, np.array([[60, 90], [20, 60]])),
    ]


def _movie(n_frames):
    rng = np.random.default_rng(0)
    return RawMovie(data=rng.random((IMG, IMG, n_frames)))


# --------------------------------------------------------------------------- #
# SingleProjectionCalibrator
# --------------------------------------------------------------------------- #

def test_single_projection_returns_calibration():
    cal = SingleProjectionCalibrator(
        layout=_Layout(_good_specs()), aligner=_ScriptedAligner([0.08]),
    ).calibrate(_movie(50))
    assert cal.accepted
    assert set(cal.transforms) == {"red"}
    assert set(cal.limits) == {"green", "red"}
    assert np.isfinite(cal.score)


def test_single_projection_flags_rejected_without_raising():
    # Gating is enabled only when acceptance criteria are supplied.
    cal = SingleProjectionCalibrator(
        layout=_Layout(_tiny_specs()), aligner=_ScriptedAligner([0.08]),
        criteria=AcceptanceCriteria(),
    ).calibrate(_movie(50))
    assert not cal.accepted
    assert cal.reasons  # explains why


# --------------------------------------------------------------------------- #
# RobustCalibrator
# --------------------------------------------------------------------------- #

def test_robust_picks_best_scoring_candidate():
    aligner = _ScriptedAligner([0.02, 0.09, 0.05])  # 3 chunks -> best is 0.09
    cal = RobustCalibrator(
        layout=_Layout(_good_specs()), aligner=aligner,
        chunk_size=100, min_candidates=3, max_trials=10,
    ).calibrate(_movie(300))
    # score is the acceptance score; the winning chunk is the one with peak 0.09
    best_expected = RobustCalibrator(
        layout=_Layout(_good_specs()), aligner=_ScriptedAligner([0.09]),
        chunk_size=100, min_candidates=1, max_trials=1,
    ).calibrate(_movie(100))
    assert cal.score == pytest.approx(best_expected.score)


def test_robust_stops_after_min_candidates():
    aligner = _ScriptedAligner([0.05, 0.06, 0.07, 0.08, 0.09])
    RobustCalibrator(
        layout=_Layout(_good_specs()), aligner=aligner,
        chunk_size=100, min_candidates=2, max_trials=10,
    ).calibrate(_movie(500))
    assert aligner.i == 2  # stopped after 2 passing candidates, didn't scan all 5


def test_robust_raises_when_nothing_passes():
    with pytest.raises(AlignmentNotFoundError):
        RobustCalibrator(
            layout=_Layout(_tiny_specs()), aligner=_ScriptedAligner([0.08]),
            chunk_size=100, min_candidates=3, max_trials=5,
        ).calibrate(_movie(500))


def test_robust_falls_back_to_single_for_short_movie():
    # Fewer frames than chunk_size -> single projection over all frames.
    aligner = _ScriptedAligner([0.08])
    cal = RobustCalibrator(
        layout=_Layout(_good_specs()), aligner=aligner,
        chunk_size=400, min_candidates=10, max_trials=30,
    ).calibrate(_movie(100))
    assert cal.accepted
    assert aligner.i == 1  # only one projection was aligned
