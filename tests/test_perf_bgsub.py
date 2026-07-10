"""Tests for the background-subtraction performance affordances (Phase 1).

The windowed-bgsub experiment was reverted after profiling showed it did not help
on real data (channel crops pad thinly, so there is little empty border to skip,
and the content scan + full-size allocation outweigh the savings). What remains is
the one-time warning steering users onto the much faster OpenCV backend, which
profiling showed to be the real single-core win (~5-20x on the dominant step).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")

import optomerge._kernels.processing as proc


def _reset_warn_flag():
    proc._WARNED_NO_CV2 = False


def test_warns_once_when_cv2_missing(monkeypatch):
    """Auto backend without cv2 warns exactly once per process, on a real movie."""
    monkeypatch.setattr(proc, "_HAS_CV2", False)
    _reset_warn_flag()
    movie = np.zeros((16, 16, 4))
    movie[4:12, 4:12, :] = 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        proc.subtract_background(movie, radius=3, backend="auto")
        proc.subtract_background(movie, radius=3, backend="auto")  # second call: silent
    msgs = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(msgs) == 1
    assert "OpenCV" in str(msgs[0].message)


def test_no_warning_when_backend_explicitly_scipy(monkeypatch):
    """Choosing scipy deliberately must not nag."""
    monkeypatch.setattr(proc, "_HAS_CV2", False)
    _reset_warn_flag()
    movie = np.zeros((16, 16, 4))
    movie[4:12, 4:12, :] = 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        proc.subtract_background(movie, radius=3, backend="scipy")
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_no_warning_on_single_frame(monkeypatch):
    """A single 2-D frame (e.g. a projection) is cheap; don't warn on it."""
    monkeypatch.setattr(proc, "_HAS_CV2", False)
    _reset_warn_flag()
    frame = np.zeros((16, 16))
    frame[4:12, 4:12] = 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        proc.subtract_background(frame, radius=3, backend="auto")
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
