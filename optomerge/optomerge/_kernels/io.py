"""
optomerge.io
============
TIFF movie I/O.

Python equivalent of ``getImageData.m`` and ``saveastiff.m``.

Array convention
----------------
Movies are stored as ``float64`` NumPy arrays with shape ``(H, W, N)`` where
``N`` is the number of frames – matching MATLAB's ``image(:,:,frame)`` layout.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_frames(
    filename: str | Path,
    frame_min: int = 0,
    frame_max: Optional[int] = None,
) -> np.ndarray:
    """Load a contiguous range of frames from a multi-page TIFF file.

    Equivalent to MATLAB ``getImageData(filename, 'tif', framemin, framemax)``.

    Parameters
    ----------
    filename : str or Path
        Path to a TIFF movie file.
    frame_min : int, optional
        First frame to load (0-indexed).  Default ``0``.
    frame_max : int or None, optional
        Last frame to load, *inclusive* (0-indexed).  ``None`` means load all
        frames through the end of the file.

    Returns
    -------
    np.ndarray
        Shape ``(H, W, N)`` float64 array.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"TIFF file not found: {filename}")

    with tifffile.TiffFile(str(filename)) as tif:
        n_pages = len(tif.pages)
        if frame_max is None:
            frame_max = n_pages - 1
        frame_max = min(frame_max, n_pages - 1)

        if frame_min < 0 or frame_min > frame_max:
            raise ValueError(
                f"Invalid frame range [{frame_min}, {frame_max}] "
                f"for file with {n_pages} pages."
            )

        # Load the requested pages
        frames = []
        for idx in range(frame_min, frame_max + 1):
            frames.append(tif.pages[idx].asarray())

    stack = np.stack(frames, axis=-1).astype(np.float64)  # (H, W, N)
    return stack


def load_movie(filename: str | Path) -> np.ndarray:
    """Load an entire TIFF movie.

    Parameters
    ----------
    filename : str or Path

    Returns
    -------
    np.ndarray, shape ``(H, W, N)``
    """
    filename = Path(filename)
    with tifffile.TiffFile(str(filename)) as tif:
        arr = tif.asarray()  # shape depends on file layout

    # Normalise to (H, W, N)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    elif arr.ndim == 3:
        # tifffile returns (N, H, W) for multi-page TIFFs
        arr = np.moveaxis(arr, 0, -1)
    else:
        warnings.warn(
            f"Unexpected array shape {arr.shape} from tifffile; "
            "assuming first axis is frames."
        )
        arr = np.moveaxis(arr, 0, -1)

    return arr.astype(np.float64)


def get_movie_info(filename: str | Path) -> Tuple[int, int, int]:
    """Return ``(height, width, n_frames)`` without loading pixel data.

    Parameters
    ----------
    filename : str or Path

    Returns
    -------
    tuple of (height, width, n_frames)
    """
    filename = Path(filename)
    with tifffile.TiffFile(str(filename)) as tif:
        first = tif.pages[0]
        h = first.shape[0]
        w = first.shape[1]
        n = len(tif.pages)
    return h, w, n


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_tiff(
    filename: str | Path,
    data: np.ndarray,
    photometric: str = "minisblack",
) -> None:
    """Save a numpy array as a multi-page TIFF.

    Parameters
    ----------
    filename : str or Path
    data : np.ndarray
        Accepted shapes:

        * ``(H, W)``       – single 2-D image
        * ``(H, W, N)``    – greyscale movie with N frames
        * ``(H, W, C, N)`` – multi-channel movie (C channels, N frames)

        Data are cast to *uint16* before writing.  Values outside [0, 65535]
        are clipped.
    photometric : str, optional
        ``'minisblack'`` (default) or ``'rgb'``.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    arr = np.clip(data, 0, 65535).astype(np.uint16)

    if arr.ndim == 2:
        tifffile.imwrite(str(filename), arr, photometric=photometric)
    elif arr.ndim == 3:
        # (H, W, N) → tifffile wants (N, H, W)
        arr_out = np.moveaxis(arr, -1, 0)
        tifffile.imwrite(str(filename), arr_out, photometric=photometric)
    elif arr.ndim == 4:
        # (H, W, C, N) → write as (N, H, W, C) for RGB-like TIFFs
        arr_out = np.moveaxis(arr, -1, 0)   # (N, H, W, C)
        tifffile.imwrite(str(filename), arr_out, photometric=photometric)
    else:
        raise ValueError(f"Cannot save array with shape {arr.shape}.")


def save_rgb_tiff(filename: str | Path, rgb_movie: np.ndarray, bit_depth: int = 8) -> None:
    """Save an RGB movie as a colour multi-page TIFF.

    Parameters
    ----------
    filename : str or Path
    rgb_movie : np.ndarray
        Shape ``(H, W, 3, N)`` with values in ``[0, 1]``.
    bit_depth : int
        8 (default) writes a uint8 RGB TIFF -- compact and shown at a fixed
        0-255 range by viewers (matches the MATLAB reference); 16 writes uint16
        (more dynamic range, but opens as an auto-contrasted composite in
        ImageJ/Fiji).
    """
    if rgb_movie.ndim != 4 or rgb_movie.shape[2] != 3:
        raise ValueError(
            f"Expected shape (H, W, 3, N), got {rgb_movie.shape}."
        )
    if bit_depth == 8:
        scaled = (np.clip(rgb_movie, 0, 1) * 255).astype(np.uint8)
    elif bit_depth == 16:
        scaled = (np.clip(rgb_movie, 0, 1) * 65535).astype(np.uint16)
    else:
        raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}.")
    # (H, W, 3, N) → (N, H, W, 3)
    out = np.moveaxis(scaled, -1, 0)
    tifffile.imwrite(str(filename), out, photometric="rgb")
