"""Tests for the constrained phase-correlation peak search.

The _peak_index logic is pure numpy (no scipy), so the core behaviour is tested
here directly. The full calculate_alignment path is exercised end-to-end by
test_oop.test_pipeline_end_to_end in an environment with scipy installed.
"""
from __future__ import annotations

import numpy as np

from optomerge._kernels.registration import _peak_index


def _map_with_peaks(shape, peaks):
    """Build a phase map with given {(row, col): value} peaks (rest ~0)."""
    m = np.zeros(shape)
    for (r, c), v in peaks.items():
        m[r, c] = v
    return m


def test_unconstrained_takes_global_max():
    m = _map_with_peaks((64, 64), {(2, 3): 0.5, (30, 30): 0.9})
    assert _peak_index(m, max_shift=0) == (30, 30)


def test_constrained_ignores_far_peak():
    m = _map_with_peaks((64, 64), {(2, 3): 0.5, (30, 30): 0.9})
    # The far, stronger peak at (30, 30) is outside the +/-5 window -> excluded.
    assert _peak_index(m, max_shift=5) == (2, 3)


def test_constrained_includes_negative_shift_via_wrap():
    # A shift of (-2, -3) appears at (H-2, W-3) = (62, 61) in FFT coords.
    m = _map_with_peaks((64, 64), {(62, 61): 0.9, (10, 10): 0.4})
    assert _peak_index(m, max_shift=5) == (62, 61)


def test_constrained_excludes_edge_beyond_window():
    # Peak at (40, 40): unwrapped shift is (40-64, 40-64) = (-24, -24), outside +/-5.
    m = _map_with_peaks((64, 64), {(40, 40): 0.9, (1, 1): 0.3})
    assert _peak_index(m, max_shift=5) == (1, 1)


def test_tiny_window_falls_back_to_global():
    # A sub-pixel window still keeps the origin cell (shift 0,0) available.
    m = _map_with_peaks((64, 64), {(0, 0): 0.2, (30, 30): 0.9})
    assert _peak_index(m, max_shift=0.5) == (0, 0)


def test_aligner_carries_max_shift():
    from optomerge import PhaseCorrelationAligner
    a = PhaseCorrelationAligner(max_shift=30)
    assert a.max_shift == 30

    from optomerge import Settings
    s = Settings.from_sources({"alignment": {"max_shift": 25.0}})
    assert s.build_aligner().max_shift == 25.0
