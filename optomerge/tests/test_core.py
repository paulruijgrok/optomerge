"""
Basic unit tests for the optomerge package.

Run with:  pytest tests/
Requires:  numpy, scipy, scikit-image, tifffile
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# transform.py
# ---------------------------------------------------------------------------

class TestZeroPadImages:
    def test_output_shape_is_power_of_2(self):
        from optomerge.transform import zero_pad_images
        a = np.ones((100, 80))
        b = np.ones((60, 90))
        na, nb = zero_pad_images(a, b)
        assert na.shape == nb.shape
        H, W = na.shape
        assert H == 2 ** int(np.ceil(np.log2(H)))
        assert W == 2 ** int(np.ceil(np.log2(W)))

    def test_identity_power_of_2_still_pads(self):
        """A 128×128 image should be padded to 256×256."""
        from optomerge.transform import zero_pad_images
        a = np.eye(128)
        b = np.eye(128)
        na, nb = zero_pad_images(a, b)
        assert na.shape[0] == 256

    def test_content_preserved(self):
        from optomerge.transform import zero_pad_images
        rng = np.random.default_rng(0)
        a = rng.random((50, 60))
        b = rng.random((40, 55))
        na, nb = zero_pad_images(a, b)
        # Content from a must appear somewhere in na
        r_off = (na.shape[0] - 50) // 2
        c_off = (na.shape[1] - 60) // 2
        np.testing.assert_allclose(na[r_off:r_off+50, c_off:c_off+60], a)

    def test_3d_movies(self):
        from optomerge.transform import zero_pad_images
        a = np.ones((50, 60, 10))
        b = np.ones((40, 55, 10))
        na, nb = zero_pad_images(a, b)
        assert na.shape == nb.shape
        assert na.shape[2] == 10


class TestTransformImage:
    def test_identity_transform(self):
        from optomerge.transform import transform_image
        rng = np.random.default_rng(1)
        im = rng.random((64, 64))
        out = transform_image(im, rot=0.0, sx=1.0, sy=1.0, tx=0.0, ty=0.0)
        # Interior pixels should be (nearly) unchanged
        np.testing.assert_allclose(out[10:-10, 10:-10], im[10:-10, 10:-10], atol=1e-6)

    def test_output_shape_preserved(self):
        from optomerge.transform import transform_image
        im = np.zeros((32, 48, 5))
        out = transform_image(im, 0.1, 1.0, 1.0)
        assert out.shape == (32, 48, 5)

    def test_nan_replaced_with_zero(self):
        from optomerge.transform import transform_image
        im = np.ones((32, 32))
        out = transform_image(im, 0.0, 1.0, 1.0, tx=100)  # large translation → out-of-bounds
        assert not np.any(np.isnan(out))


# ---------------------------------------------------------------------------
# processing.py
# ---------------------------------------------------------------------------

class TestNormImage:
    def test_values_clipped_to_unit_range(self):
        from optomerge.processing import norm_image
        arr = np.array([-10., 0., 50., 100., 200.])
        out = norm_image(arr, 0., 100.)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_min_maps_to_0(self):
        from optomerge.processing import norm_image
        arr = np.array([10., 50., 90.])
        out = norm_image(arr, 10., 90.)
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.5)

    def test_equal_min_max(self):
        from optomerge.processing import norm_image
        arr = np.ones((5, 5))
        out = norm_image(arr, 1.0, 1.0)
        assert np.all(out == 0.0)


class TestCropChannel:
    def test_basic_crop(self):
        from optomerge.processing import crop_channel
        im = np.arange(100.0).reshape(10, 10)
        bounds = np.array([[2, 6], [3, 7]])
        scrub  = np.zeros((10, 10), dtype=bool)
        out = crop_channel(im, bounds, scrub)
        assert out.shape == (5, 5)  # rows 2..6 inclusive = 5, cols 3..7 = 5

    def test_scrub_sets_to_min_nonscrub(self):
        from optomerge.processing import crop_channel
        im = np.ones((10, 10)) * 5.0
        im[0, 0] = 100.0
        bounds = np.array([[0, 9], [0, 9]])
        scrub  = np.zeros((10, 10), dtype=bool)
        scrub[0, 0] = True
        out = crop_channel(im, bounds, scrub)
        # The scrubbed pixel should have been set to min non-scrub value (5.0)
        assert out[0, 0] == pytest.approx(5.0)

    def test_movie_stack(self):
        from optomerge.processing import crop_channel
        mov = np.random.default_rng(0).random((20, 20, 7))
        bounds = np.array([[3, 12], [4, 14]])
        scrub  = np.zeros((20, 20), dtype=bool)
        out = crop_channel(mov, bounds, scrub)
        assert out.shape == (10, 11, 7)


# ---------------------------------------------------------------------------
# registration.py
# ---------------------------------------------------------------------------

class TestCalcFFT2dAlign:
    def test_identical_images_zero_offset(self):
        from optomerge.registration import _calc_fft2d_align
        rng = np.random.default_rng(42)
        im = rng.random((64, 64))
        score, offset = _calc_fft2d_align(im, im)
        # Offset should be near zero (±1 pixel tolerance for sub-pixel noise)
        assert abs(offset[0]) < 1.5
        assert abs(offset[1]) < 1.5
        assert score > 0.5

    def test_shifted_image_recovers_offset(self):
        from optomerge.registration import _calc_fft2d_align
        rng = np.random.default_rng(7)
        im = rng.random((128, 128))
        shifted = np.roll(im, 8, axis=0)
        score, offset = _calc_fft2d_align(im, shifted)
        # Roll of +8 rows means im2 is shifted down: peak at row≈-8 or 120
        # After centring: offset[0] ≈ -8
        assert abs(abs(offset[0]) - 8) < 1.5, f"offset={offset}"


class TestCalculateAlignment:
    def test_identity_alignment(self):
        from optomerge.registration import calculate_alignment
        rng = np.random.default_rng(3)
        im = rng.random((64, 64))
        t1, t2, rot, s1, s2, score = calculate_alignment(im, im)
        assert abs(t1) < 2 and abs(t2) < 2
        assert abs(rot) < 0.01
        assert abs(s1 - 1.0) < 0.02
        assert abs(s2 - 1.0) < 0.02


# ---------------------------------------------------------------------------
# segmentation.py
# ---------------------------------------------------------------------------

class TestFindChannelBounds:
    def _make_two_channel_image(self):
        """Synthetic 200×200 image with bright channel on top, dim on bottom."""
        im = np.zeros((200, 200))
        im[20:90, 10:190]   = 1.0   # top channel (bright)
        im[110:180, 10:190] = 0.4   # bottom channel (dim)
        return im

    def test_two_channels_detected(self):
        from optomerge.segmentation import find_channel_bounds
        im = self._make_two_channel_image()
        b1, s1, b2, s2 = find_channel_bounds(im, mean_projection=im)
        assert b2 is not None, "Second channel not found"
        assert s2 is not None

    def test_single_channel_with_manual_bounds(self):
        from optomerge.segmentation import find_channel_bounds
        im = np.zeros((100, 100))
        im[10:80, 5:95] = 1.0
        cb = np.array([[5, 10, 95, 80]])  # [x_left, y_top, x_right, y_bottom]
        b1, s1, b2, s2 = find_channel_bounds(
            im, mean_projection=im, channel_bounds=cb
        )
        assert b2 is None
        assert b1[0, 0] == 10   # row_start
        assert b1[0, 1] == 80   # row_end

    def test_scrub_covers_full_image(self):
        from optomerge.segmentation import find_channel_bounds
        im = self._make_two_channel_image()
        b1, s1, _, _ = find_channel_bounds(im, mean_projection=im)
        assert s1.shape == im.shape
