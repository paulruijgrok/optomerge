"""Tests for the acceptance criteria + scoring (no heavy deps needed at runtime)."""
from __future__ import annotations

import numpy as np

from optomerge import (
    AcceptanceCriteria,
    ChannelSpec,
    Settings,
    Transform,
    evaluate_acceptance,
)

IMAGE_SHAPE = (512, 512)


def _two_channel_specs():
    green = ChannelSpec(name="green", color="green",
                        bounds=np.array([[10, 245], [20, 480]]), mask=None, reference=True)
    red = ChannelSpec(name="red", color="red",
                      bounds=np.array([[266, 500], [20, 480]]), mask=None, reference=False)
    return [green, red]


def _good_transform(**kw):
    base = dict(t1=2.0, t2=-1.0, rot=np.radians(0.5), s1=1.001, s2=0.999, score=0.08)
    base.update(kw)
    return Transform(**base)


def test_good_alignment_passes():
    res = evaluate_acceptance(_two_channel_specs(), {"red": _good_transform()}, IMAGE_SHAPE)
    assert res.passed
    assert res.reasons == []
    assert np.isfinite(res.score) and res.score > 0
    assert bool(res) is True


def test_channel_too_small_fails_on_size():
    specs = _two_channel_specs()
    specs[0].bounds = np.array([[10, 110], [20, 480]])   # only 100 px tall
    res = evaluate_acceptance(specs, {"red": _good_transform()}, IMAGE_SHAPE)
    assert not res.passed
    assert any("too short" in r for r in res.reasons)


def test_large_translation_fails():
    res = evaluate_acceptance(_two_channel_specs(), {"red": _good_transform(t1=40.0)}, IMAGE_SHAPE)
    assert not res.passed
    assert any("translation" in r for r in res.reasons)


def test_large_rotation_fails():
    res = evaluate_acceptance(_two_channel_specs(),
                              {"red": _good_transform(rot=np.radians(5.0))}, IMAGE_SHAPE)
    assert not res.passed
    assert any("rotation" in r for r in res.reasons)


def test_large_scale_fails():
    res = evaluate_acceptance(_two_channel_specs(),
                              {"red": _good_transform(s1=1.05)}, IMAGE_SHAPE)
    assert not res.passed
    assert any("scale" in r for r in res.reasons)


def test_single_channel_passes_size_only():
    green = ChannelSpec(name="green", color="green",
                        bounds=np.array([[10, 500], [10, 500]]), mask=None, reference=True)
    res = evaluate_acceptance([green], {}, IMAGE_SHAPE)
    assert res.passed
    assert np.isfinite(res.score)


def test_no_channels_fails():
    res = evaluate_acceptance([], {}, IMAGE_SHAPE)
    assert not res.passed
    assert np.isnan(res.score)


def test_score_rewards_bigger_channel_and_higher_peak():
    specs = _two_channel_specs()
    small = specs[0].bounds.copy()
    low = evaluate_acceptance(specs, {"red": _good_transform(score=0.02)}, IMAGE_SHAPE).score
    high = evaluate_acceptance(specs, {"red": _good_transform(score=0.09)}, IMAGE_SHAPE).score
    assert high > low  # higher cross-correlation peak scores higher

    specs[0].bounds = np.array([[10, 120], [20, 200]])   # much smaller channel
    smaller = evaluate_acceptance(specs, {"red": _good_transform(score=0.09)}, IMAGE_SHAPE).score
    specs[0].bounds = small
    bigger = evaluate_acceptance(specs, {"red": _good_transform(score=0.09)}, IMAGE_SHAPE).score
    assert bigger > smaller


def test_criteria_are_configurable():
    # A translation that fails the default should pass a looser criterion.
    loose = AcceptanceCriteria(max_translation_px=100.0)
    res = evaluate_acceptance(_two_channel_specs(), {"red": _good_transform(t1=40.0)},
                              IMAGE_SHAPE, criteria=loose)
    assert res.passed


def test_build_acceptance_from_settings_maps_fields():
    s = Settings.from_sources({"acceptance": {"max_rotation_deg": 5.0, "min_size_row_frac": 0.5}})
    crit = s.build_acceptance()
    assert crit.max_rotation_deg == 5.0
    assert crit.min_size_frac == (0.5, 0.85)
