"""Tests for RGB TIFF output bit depth."""
from __future__ import annotations

import numpy as np
import pytest

from optomerge._kernels import io as _io


class _Recorder:
    """Stand-in for tifffile.imwrite that records the written array."""
    def __init__(self):
        self.arr = None

    def imwrite(self, path, arr, **kw):
        self.arr = np.asarray(arr)


def _rgb():
    # (H, W, 3, N) in [0,1] with a mid-grey value so the scaling is checkable.
    return np.full((4, 4, 3, 2), 0.5)


def test_default_is_8bit(monkeypatch, tmp_path):
    rec = _Recorder()
    monkeypatch.setattr(_io, "tifffile", rec)
    _io.save_rgb_tiff(tmp_path / "out.tif", _rgb())          # default bit_depth
    assert rec.arr.dtype == np.uint8
    assert rec.arr.max() == 127                               # 0.5 * 255


def test_16bit_option(monkeypatch, tmp_path):
    rec = _Recorder()
    monkeypatch.setattr(_io, "tifffile", rec)
    _io.save_rgb_tiff(tmp_path / "out.tif", _rgb(), bit_depth=16)
    assert rec.arr.dtype == np.uint16
    assert rec.arr.max() == 32767                             # 0.5 * 65535


def test_bad_bit_depth_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_io, "tifffile", _Recorder())
    with pytest.raises(ValueError):
        _io.save_rgb_tiff(tmp_path / "out.tif", _rgb(), bit_depth=12)


def test_config_and_pipeline_default_8bit():
    from optomerge import Settings, MergePipeline
    assert Settings().io.rgb_bitdepth == 8
    assert MergePipeline("m.tif").rgb_bitdepth == 8
    assert Settings.from_sources({"io": {"rgb_bitdepth": 16}}).io.rgb_bitdepth == 16


def test_tiffwriter_carries_bit_depth():
    from optomerge import TiffWriter, MovieWriter
    assert TiffWriter().bit_depth == 8
    assert MovieWriter.for_path("x.tif", bit_depth=16).bit_depth == 16
