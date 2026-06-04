"""
optomerge._gpu
==============
GPU-accelerated variants of the core processing functions (CuPy sketch).

This module is **optional**: it will import cleanly even when CuPy is not
installed.  All public functions fall back to their CPU equivalents when no
GPU is available.

Requirements
------------
* NVIDIA GPU with CUDA toolkit installed.
* CuPy matching your CUDA version::

    pip install cupy-cuda12x   # for CUDA 12.x
    pip install cupy-cuda11x   # for CUDA 11.x

Usage
-----
.. code-block:: python

    from optomerge._gpu import gpu_available, subtract_background_gpu, transform_image_gpu

    if gpu_available():
        result = subtract_background_gpu(movie, radius=10)
    else:
        from optomerge.processing import subtract_background
        result = subtract_background(movie, radius=10)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# GPU availability probe
# ---------------------------------------------------------------------------

try:
    import cupy as cp  # type: ignore
    import cupyx.scipy.ndimage as cpnd  # type: ignore
    HAS_GPU: bool = True
except ImportError:
    HAS_GPU = False


def gpu_available() -> bool:
    """Return ``True`` if CuPy is installed and a CUDA device is accessible."""
    if not HAS_GPU:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()  # raises if no device
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# GPU subtract_background (sketch — requires CuPy)
# ---------------------------------------------------------------------------

def subtract_background_gpu(
    movie: np.ndarray,
    radius: int = 10,
    method: str = "morph_open",
    vram_gb: float = 4.0,
) -> np.ndarray:
    """GPU-accelerated background subtraction.

    Falls back transparently to the CPU implementation when CuPy is not
    available or no GPU device is found.

    Parameters
    ----------
    movie : np.ndarray
        Shape ``(H, W, N)``, dtype float64.
    radius : int
        Background radius in pixels.
    method : str
        ``'morph_open'`` or ``'gaussian'``.
    vram_gb : float
        Approximate usable VRAM in GB.  Frames are processed in chunks that
        fit within ``0.7 × vram_gb``.

    Returns
    -------
    np.ndarray, same shape, dtype float64.
    """
    if not gpu_available():
        # Transparent CPU fallback
        from .processing import subtract_background
        return subtract_background(movie, radius=radius, method=method)

    movie = np.asarray(movie, dtype=np.float64)
    single_frame = movie.ndim == 2
    if single_frame:
        movie = movie[:, :, np.newaxis]

    H, W, N = movie.shape
    bytes_per_frame = H * W * 8  # float64 = 8 bytes
    chunk_max = max(1, int(vram_gb * 0.7 * 1e9 / bytes_per_frame))

    result = np.zeros_like(movie)

    if method == "morph_open":
        from skimage.morphology import disk as _disk
        se_np = _disk(radius).astype(np.float64)
        se_gpu = cp.asarray(se_np)

        for start in range(0, N, chunk_max):
            end = min(start + chunk_max, N)
            chunk_gpu = cp.asarray(movie[:, :, start:end])
            out_gpu = cp.zeros_like(chunk_gpu)
            for i in range(end - start):
                bg = cpnd.grey_opening(chunk_gpu[:, :, i], footprint=se_gpu)
                out_gpu[:, :, i] = chunk_gpu[:, :, i] - bg
            result[:, :, start:end] = cp.asnumpy(out_gpu)

    elif method == "gaussian":
        for start in range(0, N, chunk_max):
            end = min(start + chunk_max, N)
            chunk_gpu = cp.asarray(movie[:, :, start:end])
            out_gpu = cp.zeros_like(chunk_gpu)
            for i in range(end - start):
                bg = cpnd.gaussian_filter(chunk_gpu[:, :, i], sigma=float(radius))
                out_gpu[:, :, i] = chunk_gpu[:, :, i] - bg
            result[:, :, start:end] = cp.asnumpy(out_gpu)

    else:
        raise ValueError(f"Unknown method: '{method}'")

    if single_frame:
        result = result[:, :, 0]
    return result


# ---------------------------------------------------------------------------
# GPU transform_image (sketch — requires CuPy)
# ---------------------------------------------------------------------------

def transform_image_gpu(
    im: np.ndarray,
    rot: float,
    sx: float,
    sy: float,
    tx: float = 0.0,
    ty: float = 0.0,
    vram_gb: float = 4.0,
) -> np.ndarray:
    """GPU-accelerated affine image transformation.

    Falls back transparently to the CPU implementation when CuPy is not
    available or no GPU device is found.

    Parameters
    ----------
    im : np.ndarray
        Shape ``(H, W, N)``, dtype float64.
    rot, sx, sy, tx, ty : float
        Affine transform parameters (same convention as :func:`transform_image`).
    vram_gb : float
        Approximate usable VRAM in GB for chunked processing.

    Returns
    -------
    np.ndarray, same shape, dtype float64.
    """
    if not gpu_available():
        from .transform import transform_image
        return transform_image(im, rot, sx, sy, tx, ty)

    im = np.asarray(im, dtype=np.float64)
    single_frame = im.ndim == 2
    if single_frame:
        im = im[:, :, np.newaxis]

    H, W, N = im.shape
    bytes_per_frame = H * W * 8
    chunk_max = max(1, int(vram_gb * 0.7 * 1e9 / bytes_per_frame))

    # Build coordinate map on CPU (cheap), then reuse on GPU
    range_row = np.arange(1, H + 1) - H / 2 - 0.5
    range_col = np.arange(1, W + 1) - W / 2 - 0.5
    meshx, meshy = np.meshgrid(range_col, range_row)

    x_flat = meshx.ravel()
    y_flat = meshy.ravel()
    ones = np.ones(H * W, dtype=np.float64)
    coords = np.stack([x_flat, y_flat, ones], axis=0)

    tmat = np.array([
        [np.cos(rot) * sx, -np.sin(rot) * sy, tx],
        [np.sin(rot) * sx,  np.cos(rot) * sy, ty],
        [0.0,               0.0,               1.0],
    ])
    ncoords = tmat @ coords

    new_x = ncoords[0].reshape(H, W)
    new_y = ncoords[1].reshape(H, W)
    row_idx = (new_y + H / 2 - 0.5).ravel()
    col_idx = (new_x + W / 2 - 0.5).ravel()
    coords_gpu = cp.asarray(np.stack([row_idx, col_idx], axis=0))

    result = np.zeros_like(im)

    for start in range(0, N, chunk_max):
        end = min(start + chunk_max, N)
        chunk_gpu = cp.asarray(im[:, :, start:end])
        out_gpu = cp.zeros_like(chunk_gpu)
        for i in range(end - start):
            interp = cpnd.map_coordinates(
                chunk_gpu[:, :, i], coords_gpu,
                order=3, mode="constant", cval=0.0, prefilter=True,
            )
            out_gpu[:, :, i] = interp.reshape(H, W)
        result[:, :, start:end] = cp.asnumpy(out_gpu)

    result[np.isnan(result)] = 0.0

    if single_frame:
        result = result[:, :, 0]
    return result
