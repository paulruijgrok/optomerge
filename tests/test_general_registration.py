"""Tests for the general-purpose aligners (ECC, mutual information, keypoint).

The strategy is a round-trip: take a richly textured reference, warp it by a
*known* transform to synthesise the moving channel, and check each aligner
recovers the inverse -- i.e. the transform that, applied to the moving channel,
lands it back on the reference (the same sign convention the pipeline uses).
"""

from __future__ import annotations

import numpy as np
import pytest

from optomerge.channel import Channel
from optomerge.transform import Transform
from optomerge._kernels.transform import _build_cv2_matrix
from optomerge.general_registration import (
    EccAligner,
    FeatureMatchAligner,
    MutualInformationAligner,
)

H = W = 180


def _texture(seed: int = 3) -> np.ndarray:
    """A blobby, corner-rich image both intensity and keypoint methods can use."""
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:H, :W]
    img = np.zeros((H, W), dtype=np.float64)
    for _ in range(60):
        y, x = rng.integers(20, H - 20), rng.integers(20, W - 20)
        r = rng.integers(3, 9)
        img += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * r * r)) * rng.uniform(0.5, 1.0)
    return (img - img.min()) / np.ptp(img)


def _channel(arr: np.ndarray, reference: bool = False) -> Channel:
    return Channel(arr, "c", [[0, H - 1], [0, W - 1]], reference=reference)


def _moving(img: np.ndarray, gx: float, gy: float, rot: float = 0.0, s: float = 1.0) -> np.ndarray:
    """Warp the reference by a known transform to make the moving channel."""
    return Transform(t1=gx, t2=gy, rot=rot, s1=s, s2=s).apply(img)


# --------------------------------------------------------------------------- #
#  Transform.from_affine_matrix: exact inverse of the pipeline's own builder
# --------------------------------------------------------------------------- #

def test_from_affine_matrix_round_trips_build_cv2_matrix():
    rng = np.random.default_rng(0)
    for _ in range(500):
        h, w = int(rng.integers(20, 300)), int(rng.integers(20, 300))
        rot = rng.uniform(-0.5, 0.5)
        s1, s2 = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
        tx, ty = rng.uniform(-40, 40), rng.uniform(-40, 40)
        m = _build_cv2_matrix(h, w, rot, s1, s2, tx, ty)
        t = Transform.from_affine_matrix(m, (h, w))
        assert t.rot == pytest.approx(rot, abs=1e-9)
        assert t.s1 == pytest.approx(s1, abs=1e-9)
        assert t.s2 == pytest.approx(s2, abs=1e-9)
        assert t.t1 == pytest.approx(tx, abs=1e-9)
        assert t.t2 == pytest.approx(ty, abs=1e-9)


def test_from_affine_matrix_accepts_3x3():
    m3 = np.eye(3)
    m3[0, 2], m3[1, 2] = 5.0, -3.0  # pure pixel-coord translation
    t = Transform.from_affine_matrix(m3, (100, 100))
    assert t.t1 == pytest.approx(5.0)
    assert t.t2 == pytest.approx(-3.0)
    assert t.rot == pytest.approx(0.0)


def test_from_affine_matrix_rejects_bad_shape():
    with pytest.raises(ValueError):
        Transform.from_affine_matrix(np.zeros((2, 2)), (10, 10))


# --------------------------------------------------------------------------- #
#  Translation recovery: every method should invert a known shift
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "aligner",
    [
        EccAligner(motion="translation"),
        MutualInformationAligner(max_shift=15, step=1),
        FeatureMatchAligner(detector="orb"),
        FeatureMatchAligner(detector="sift"),
    ],
    ids=["ecc", "mi", "orb", "sift"],
)
def test_recovers_translation(aligner):
    img = _texture()
    gx, gy = 7.0, -5.0
    ref = _channel(img, reference=True)
    mov = _channel(_moving(img, gx, gy))
    t = aligner.align(ref, mov)
    # Applied to the moving channel, the fit undoes the synthetic shift.
    assert t.t1 == pytest.approx(-gx, abs=0.7)
    assert t.t2 == pytest.approx(-gy, abs=0.7)
    assert t.score > 0.0


@pytest.mark.parametrize(
    "aligner",
    [EccAligner(motion="euclidean"), FeatureMatchAligner(detector="sift", motion="similarity")],
    ids=["ecc-euclidean", "sift-similarity"],
)
def test_recovers_rotation(aligner):
    img = _texture()
    rot = np.radians(3.0)
    ref = _channel(img, reference=True)
    mov = _channel(_moving(img, 4.0, 3.0, rot=rot))
    t = aligner.align(ref, mov)
    # The recovered rotation inverts the +3 deg synthetic rotation.
    assert np.degrees(t.rot) == pytest.approx(-3.0, abs=0.6)
    assert t.score > 0.0


# --------------------------------------------------------------------------- #
#  Graceful failure: featureless input returns identity, never raises
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "aligner",
    [
        EccAligner(motion="translation", max_iterations=20),
        FeatureMatchAligner(detector="orb"),
        FeatureMatchAligner(detector="sift"),
    ],
    ids=["ecc", "orb", "sift"],
)
def test_blank_input_returns_identity(aligner):
    blank = _channel(np.zeros((H, W)), reference=True)
    blank2 = _channel(np.zeros((H, W)))
    t = aligner.align(blank, blank2)
    assert t.is_identity
    assert t.score == 0.0


def test_mi_rejects_mismatched_shapes():
    ref = _channel(np.zeros((H, W)), reference=True)
    small = Channel(np.zeros((H // 2, W // 2)), "c", [[0, H // 2 - 1], [0, W // 2 - 1]])
    t = MutualInformationAligner(max_shift=10).align(ref, small)
    assert t.is_identity


# --------------------------------------------------------------------------- #
#  max_shift guard rejects an implausibly large fit
# --------------------------------------------------------------------------- #

def test_max_shift_guard_rejects_large_translation():
    img = _texture()
    ref = _channel(img, reference=True)
    mov = _channel(_moving(img, 20.0, 0.0))  # true shift 20 px
    guarded = EccAligner(motion="translation", max_shift=5.0).align(ref, mov)
    assert guarded.is_identity  # 20 px > 5 px guard -> rejected


def test_constructors_validate_arguments():
    with pytest.raises(ValueError):
        EccAligner(motion="projective")
    with pytest.raises(ValueError):
        FeatureMatchAligner(detector="surf")
    with pytest.raises(ValueError):
        FeatureMatchAligner(motion="rigid")
    with pytest.raises(ValueError):
        MutualInformationAligner(max_shift=0)


def test_ecc_on_unequal_crops_gives_the_merge_canvas_transform():
    """ECC needs equal shapes: unequal crops are trimmed, fitted, then converted.

    Both crops are cut from one smooth texture T: green = T[10:50, 10:70] (40x60),
    red = T[13:46, 12:70] (33x58). Centred in the 64x64 merge canvas, green pixel
    (R, C) is T[R-2, C+8] and red is T[R-2, C+9], so the merge needs red moved one
    column right: t = (-1, 0). Before the trim ECC returned identity here.
    """
    pytest.importorskip("cv2")
    scipy_ndimage = pytest.importorskip("scipy.ndimage")
    from optomerge.calibration import fit_in_canvas
    from optomerge.channel import Channel

    rng = np.random.default_rng(1)
    tex = scipy_ndimage.gaussian_filter(rng.random((100, 100)), 3.0)
    green = Channel(tex[10:50, 10:70], "green", np.array([[0, 39], [0, 59]]), reference=True)
    red = Channel(tex[13:46, 12:70], "red", np.array([[0, 32], [0, 57]]))
    for c in (green, red):
        c.calibrate()

    _, t = fit_in_canvas(EccAligner(motion="translation"), green, red, [green, red])
    assert t.score > 0.9
    assert (t.t1, t.t2) == pytest.approx((-1.0, 0.0), abs=0.1)
