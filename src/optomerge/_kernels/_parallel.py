"""Process-wide default for the per-frame worker-thread count.

The per-frame kernels (background subtraction, affine transform) thread over
frames, defaulting ``n_workers`` to all logical CPUs. When the batch runner
processes several movies in parallel *processes*, each process should use only a
share of the cores, or the P processes x T threads oversubscribe and get slower.

This module holds a process-wide default that a worker can lower once at start-up
(``set_default_workers(cores // P)``) without threading an ``n_workers`` argument
through every kernel call site. An explicit ``n_workers=`` passed to a kernel
still wins; this only changes the ``n_workers is None`` default.
"""
from __future__ import annotations

from typing import Optional

_DEFAULT_WORKERS: Optional[int] = None  # None => use all logical CPUs


def get_default_workers() -> Optional[int]:
    """Return the process-wide default worker count (``None`` = all CPUs)."""
    return _DEFAULT_WORKERS


def set_default_workers(n: Optional[int]) -> None:
    """Set the process-wide default worker count for the per-frame kernels.

    ``None`` restores the "use all logical CPUs" default. Any integer is clamped
    to at least 1.
    """
    global _DEFAULT_WORKERS
    _DEFAULT_WORKERS = None if n is None else max(1, int(n))
