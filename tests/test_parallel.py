"""Tests for file-level parallelism in the batch runner (Phase 2).

Covers the worker-thread-count seam (so parallel processes don't oversubscribe)
and an end-to-end check that ``run_optomerge.py --workers N`` produces byte-for-
byte the same output as the sequential run -- parallelism must change only the
schedule, never the result.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import optomerge as om

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Worker-thread-count seam
# --------------------------------------------------------------------------- #

def test_default_workers_set_get_clamp():
    try:
        om.set_default_workers(3)
        assert om.get_default_workers() == 3
        om.set_default_workers(0)          # clamped to >= 1
        assert om.get_default_workers() == 1
        om.set_default_workers(None)       # restores "all CPUs"
        assert om.get_default_workers() is None
    finally:
        om.set_default_workers(None)


def test_capped_default_matches_uncapped_result():
    """Lowering the thread default must not change kernel output."""
    pytest.importorskip("scipy")
    from optomerge._kernels.processing import subtract_background
    rng = np.random.default_rng(0)
    mov = rng.random((48, 64, 12))
    ref = subtract_background(mov, radius=6, n_workers=1)
    try:
        om.set_default_workers(1)
        got = subtract_background(mov, radius=6)   # n_workers=None -> honours default
    finally:
        om.set_default_workers(None)
    np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------- #
# End-to-end: parallel output == sequential output
# --------------------------------------------------------------------------- #

def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _write_synth_movie(path: Path, n: int = 8, shift: int = 2, seed: int = 0) -> None:
    """A tiny valid 2-channel movie the pipeline accepts.

    Two stacked 32-row channels each filled with the *same* textured pattern
    (so phase correlation finds a strong peak), the bottom shifted by ``shift``
    rows. The channels fill most of their half so they pass the size-based
    acceptance criteria.
    """
    import tifffile
    rng = np.random.default_rng(seed)
    H, W = 64, 64
    base = 1.0 + 0.4 * rng.random((26, 58))
    mov = np.zeros((H, W, n), np.float32)
    for k in range(n):
        mov[3:29, 3:61, k] = base
        mov[35:61, 3:61, k] = np.roll(base, shift, axis=0)
    tifffile.imwrite(path, np.moveaxis(mov, 2, 0))   # (n, H, W)


def _run_batch(in_dir: Path, out_dir: Path, workers: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "run_optomerge.py"),
         "--input", str(in_dir), "--output", str(out_dir),
         "--channel-order", "top_green_fils_bottom_red_heads",
         "--workers", str(workers), "--overwrite"],
        capture_output=True, text=True, timeout=300,
    )


@pytest.mark.skipif(not (_have("scipy") and _have("skimage") and _have("tifffile")),
                    reason="requires scipy, scikit-image and tifffile")
def test_parallel_output_matches_sequential(tmp_path):
    import tifffile
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(3):
        _write_synth_movie(in_dir / f"movie_{i:02d}.tif", shift=1 + i, seed=i)

    seq = _run_batch(in_dir, tmp_path / "seq", workers=1)
    par = _run_batch(in_dir, tmp_path / "par", workers=2)
    assert seq.returncode == 0, seq.stderr
    assert par.returncode == 0, par.stderr

    seq_tifs = sorted((tmp_path / "seq").rglob("*_aligned.tif"))
    par_tifs = sorted((tmp_path / "par").rglob("*_aligned.tif"))
    assert len(seq_tifs) == 3 and len(par_tifs) == 3
    for s, p in zip(seq_tifs, par_tifs):
        assert s.name == p.name
        np.testing.assert_array_equal(tifffile.imread(s), tifffile.imread(p))
