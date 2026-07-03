"""
optomerge.channel
====================
Channel and ScrubImage.

A *Channel* is a crop out of the raw frame(s) carrying one kind of information
(usually one colour / one molecular species). It owns its pixels *and* the
bookkeeping needed to normalise and register it -- the bounding box, the scrub
mask, and the intensity limits -- which in the original functional code travel
as loose tuples (``bounds1, scrub1, green_min, green_max`` ...). Bundling them
is the main robustness win: it is impossible to normalise channel A with channel
B's limits by accident.

A *ScrubImage* is a derived, enlarged version of a channel used to drive
sub-pixel alignment.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._core import norm_image, subtract_background as _subtract_background
from .transform import Transform


class Channel:
    """One channel cropped from a raw movie / frame / bunch.

    Parameters
    ----------
    data : np.ndarray, shape (h, w) or (h, w, n)
        Cropped pixel data (single image or a stack across a FrameBunch). This
        is already the cropped, scrub-applied region (see :meth:`crop`).
    name : str
        Logical name, e.g. ``"green_fils"`` or ``"red_heads"``.
    bounds : np.ndarray, shape (2, 2)
        ``[[row_start, row_end], [col_start, col_end]]`` (0-indexed, inclusive)
        of the crop in the raw frame -- the convention used throughout the
        functional core.
    color : str, optional
        Output colour plane this channel maps to: ``"red"`` / ``"green"`` /
        ``"blue"``.
    mask : np.ndarray, optional
        Full-frame boolean scrub mask (``True`` = outside channel). Retained for
        provenance; it has already been applied during cropping.
    reference : bool
        Whether this is the alignment reference (others register onto it).
    """

    def __init__(
        self,
        data: np.ndarray,
        name: str,
        bounds: np.ndarray,
        color: Optional[str] = None,
        mask: Optional[np.ndarray] = None,
        reference: bool = False,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.name = name
        self.bounds = np.asarray(bounds, dtype=int)
        self.color = color
        self.mask = mask
        self.reference = reference

        # Intensity limits used for normalisation; set by calibrate().
        self.vmin: Optional[float] = None
        self.vmax: Optional[float] = None

        # Transform mapping this channel onto the reference (identity for ref).
        self.transform: Transform = Transform()

    # -- normalisation ----------------------------------------------------- #

    def calibrate(self, exclude_fraction: float = 0.0) -> "Channel":
        """Measure and store intensity limits (vmin/vmax) for normalisation.

        Replaces the free-floating ``green_min/green_max/red_min/red_max``. By
        default uses the raw min/max of the (cropped) data, matching the
        original pipeline. ``exclude_fraction`` enables robust percentile limits.
        """
        if exclude_fraction > 0:
            self.vmin = float(np.percentile(self.data, 100.0 * exclude_fraction))
            self.vmax = float(np.percentile(self.data, 100.0 * (1 - exclude_fraction)))
        else:
            self.vmin = float(self.data.min())
            self.vmax = float(self.data.max())
        return self

    def normalized(self) -> np.ndarray:
        """Return pixels scaled to [0, 1] using stored vmin/vmax."""
        if self.vmin is None or self.vmax is None:
            self.calibrate()
        return norm_image(self.data, self.vmin, self.vmax)

    def subtract_background(self, radius: int = 10) -> "Channel":
        """Return a *new* Channel with morphological background subtracted.

        Operates on normalised data (the order used by the original pipeline:
        normalise, then subtract background).
        """
        out = self._copy_meta(_subtract_background(self.normalized(), radius=radius))
        # Data is now already normalised+bg-subtracted, so freeze its limits.
        out.vmin, out.vmax = 0.0, 1.0
        return out

    # -- alignment helpers ------------------------------------------------- #

    def projection(self, how: str = "mean") -> np.ndarray:
        """Collapse a channel stack to a single 2-D image for registration."""
        if self.data.ndim == 2:
            return self.data
        if how == "mean":
            return self.data.mean(axis=2)
        if how == "max":
            return self.data.max(axis=2)
        raise ValueError(f"Unknown projection '{how}'")

    def scrub_image(self, upscale: int = 4, fast: bool = False) -> "ScrubImage":
        """Build an enlarged helper image to refine sub-pixel alignment.

        Upsamples the (normalised) channel projection by ``upscale``. ``fast``
        uses nearest-neighbour interpolation (order 0); otherwise cubic. The
        default :class:`~optomerge.registration.PhaseCorrelationAligner`
        aligns on projections directly and does not require this; it exists for
        future aligners that want finer-than-pixel resolution.
        """
        from scipy.ndimage import zoom

        proj = norm_image(self.projection("mean"), self.vmin or 0.0, self.vmax or 1.0)
        order = 0 if fast else 3
        enlarged = zoom(proj, upscale, order=order)
        return ScrubImage(enlarged, source_name=self.name, upscale=upscale)

    # -- helpers ----------------------------------------------------------- #

    def _copy_meta(self, new_data: np.ndarray) -> "Channel":
        out = Channel(
            new_data, self.name, self.bounds,
            color=self.color, mask=self.mask, reference=self.reference,
        )
        out.vmin, out.vmax = self.vmin, self.vmax
        out.transform = self.transform
        return out

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def __array__(self) -> np.ndarray:
        return self.data

    def __repr__(self) -> str:
        role = "reference" if self.reference else "moving"
        return f"Channel(name={self.name!r}, color={self.color!r}, {role}, shape={self.data.shape})"


class ScrubImage:
    """Enlarged helper derived from a :class:`Channel` to aid alignment.

    Parameters
    ----------
    data : np.ndarray
        The enlarged image.
    source_name : str
        Name of the Channel this was built from.
    upscale : int
        Linear upsampling factor relative to the source channel.
    """

    def __init__(self, data: np.ndarray, source_name: str, upscale: int) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.source_name = source_name
        self.upscale = upscale

    def to_native(self, transform: Transform) -> Transform:
        """Rescale a transform fitted in scrub space back to native pixels."""
        return transform.rescaled(1.0 / self.upscale)

    def __array__(self) -> np.ndarray:
        return self.data

    def __repr__(self) -> str:
        return f"ScrubImage(source={self.source_name!r}, upscale={self.upscale}, shape={self.data.shape})"
