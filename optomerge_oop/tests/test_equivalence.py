"""
Equivalence test: optomerge_oo vs the original optomerge.AlignmentPipeline.

Run this where scipy / scikit-image / tifffile are installed (the sandbox used
to build the package has none of them, so this is the local verification step).

    python -m pytest optomerge_oop/tests/test_equivalence.py
    # or, standalone:
    python optomerge_oop/tests/test_equivalence.py path/to/movie.tif

It asserts the OO pipeline produces a bit-comparable RGB movie to the original
functional pipeline on the same input, default settings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make both packages importable.
_HERE = Path(__file__).resolve()
_OOP_ROOT = _HERE.parents[1]                  # optomerge_oop
_FUNC_ROOT = _HERE.parents[2] / "optomerge"   # functional package root
for p in (_OOP_ROOT, _FUNC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _find_test_movie() -> Path | None:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    data_dir = _HERE.parents[2] / "test_data"
    if data_dir.is_dir():
        tifs = sorted(data_dir.rglob("*.tif")) + sorted(data_dir.rglob("*.tiff"))
        if tifs:
            return tifs[0]
    return None


def run_equivalence(movie_path: Path) -> None:
    from optomerge import AlignmentPipeline           # original
    import optomerge_oo as oo                          # new OO layer

    # --- original ---
    ref = AlignmentPipeline(str(movie_path)).run()

    # --- OO (defaults mirror the original: auto layout, phase correlation) ---
    got = oo.MergePipeline(movie_path).run().to_array()

    assert got.shape == ref.shape, f"shape mismatch: {got.shape} vs {ref.shape}"
    max_abs = float(np.max(np.abs(got - ref)))
    print(f"shapes match: {got.shape}; max |Δ| = {max_abs:.3e}")
    assert np.allclose(got, ref, atol=1e-8), f"pixel mismatch, max |Δ| = {max_abs}"
    print("EQUIVALENT: OO pipeline reproduces the original output.")


def test_equivalence():
    movie = _find_test_movie()
    if movie is None or not movie.exists():
        import pytest  # type: ignore
        pytest.skip("No test movie found.")
    run_equivalence(movie)


if __name__ == "__main__":
    m = _find_test_movie()
    if m is None or not m.exists():
        print("No test movie found. Pass a path: python test_equivalence.py movie.tif")
        sys.exit(1)
    run_equivalence(m)
