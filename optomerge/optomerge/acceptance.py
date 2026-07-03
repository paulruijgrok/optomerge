"""
optomerge.acceptance
=====================
Acceptance criteria and scoring for a detected channel layout + alignment.

Ported from the MATLAB ``alignRGB.m`` reference (``scoreChannelAlignment`` and
the ``imRel*Cutoff`` filters). The point is to *reject nonsensical results* — a
channel that came out far too small, or a transform with an implausibly large
translation / rotation / scale — before they are applied to a whole movie or
propagated across a set of movies.

Two things live here:

* :class:`AcceptanceCriteria` — the configurable thresholds + scoring weights.
* :func:`evaluate` — check a candidate (resolved specs + fitted transforms)
  against the criteria and return pass/fail, human-readable reasons, and a
  scalar score used to pick the best candidate among several.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .transform import Transform


@dataclass
class AcceptanceCriteria:
    """Thresholds for accepting a channel-detection + alignment result.

    Defaults mirror the ``imRel*Cutoff`` values in ``alignRGB.m``.

    Attributes
    ----------
    min_size_frac : (float, float)
        Minimum reference-channel ``(row, col)`` extent as a fraction of the
        raw image dimensions. The row fraction is multiplied by the number of
        detected channels (stacked channels each span ~1/N of the height), so
        two stacked half-height channels give ~1.0. From ``imRelSizeCutoff``.
    max_translation_px : float
        Maximum allowed ``|t1|`` / ``|t2|`` translation (pixels). ``imRelTransCutoff``.
    max_rotation_deg : float
        Maximum allowed ``|rotation|`` (degrees). ``imRelRotCutoff``.
    max_scale_pct : float
        Maximum allowed scale deviation ``|s - 1| * 100`` (percent). ``imRelScaleCutoff``.
    weight_channel_size, weight_cross_corr : float
        Scoring weights (channel size vs cross-correlation peak). From the
        ``weights`` struct in ``scoreChannelAlignment`` (1.0 and 3.0).
    cross_corr_norm : float
        Normalisation for the cross-correlation peak (``crossCorrelationMax``).
    """

    min_size_frac: Tuple[float, float] = (0.7, 0.85)
    max_translation_px: float = 25.0
    max_rotation_deg: float = 2.0
    max_scale_pct: float = 2.0
    weight_channel_size: float = 1.0
    weight_cross_corr: float = 3.0
    cross_corr_norm: float = 0.1


@dataclass
class AcceptanceResult:
    """Outcome of checking a candidate against :class:`AcceptanceCriteria`.

    ``passed`` is True only if every criterion held. ``reasons`` lists the
    criteria that failed (empty when ``passed``). ``score`` is the combined
    quality score (higher is better; ``nan`` if it could not be computed).
    """

    passed: bool
    score: float
    reasons: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


def _channel_extent(bounds: np.ndarray) -> Tuple[int, int]:
    """Return the ``(rows, cols)`` extent of a channel from its bounds.

    ``bounds`` is ``[[row_start, row_end], [col_start, col_end]]`` (inclusive),
    the convention used throughout the package.
    """
    b = np.asarray(bounds, dtype=float)
    rows = abs(b[0, 1] - b[0, 0])
    cols = abs(b[1, 1] - b[1, 0])
    return rows, cols


def evaluate(
    specs: Sequence,
    transforms: "Dict[str, Transform]",
    image_shape: Sequence[int],
    criteria: AcceptanceCriteria | None = None,
) -> AcceptanceResult:
    """Check a detected layout + fitted transforms against the criteria.

    Parameters
    ----------
    specs : sequence of ChannelSpec (or Channel)
        Resolved channels; each must expose ``.bounds``, ``.reference`` and
        ``.name``. Exactly one is expected to be the reference.
    transforms : dict[str, Transform]
        Fitted transform per moving channel (by name).
    image_shape : sequence of int
        Raw frame shape; the first two entries are ``(rows, cols)``.
    criteria : AcceptanceCriteria, optional
        Thresholds/weights; defaults to :class:`AcceptanceCriteria`.
    """
    criteria = criteria or AcceptanceCriteria()
    reasons: List[str] = []

    resolved = [s for s in specs if getattr(s, "bounds", None) is not None]
    n_channels = len(resolved)
    if n_channels == 0:
        return AcceptanceResult(passed=False, score=float("nan"),
                                reasons=["no channels detected"])

    img_rows, img_cols = float(image_shape[0]), float(image_shape[1])

    # Reference channel (falls back to the first resolved spec).
    reference = next((s for s in resolved if getattr(s, "reference", False)), resolved[0])
    ref_rows, ref_cols = _channel_extent(reference.bounds)

    # -- size criterion --------------------------------------------------- #
    row_frac = ref_rows / img_rows * n_channels
    col_frac = ref_cols / img_cols
    if row_frac <= criteria.min_size_frac[0]:
        reasons.append(
            f"channel too short: row fraction {row_frac:.3f} <= {criteria.min_size_frac[0]}"
        )
    if col_frac <= criteria.min_size_frac[1]:
        reasons.append(
            f"channel too narrow: col fraction {col_frac:.3f} <= {criteria.min_size_frac[1]}"
        )

    # -- transform criteria (per moving channel) -------------------------- #
    for name, t in transforms.items():
        if abs(t.t1) > criteria.max_translation_px or abs(t.t2) > criteria.max_translation_px:
            reasons.append(
                f"{name}: translation ({t.t1:.2f}, {t.t2:.2f}) px exceeds "
                f"{criteria.max_translation_px}"
            )
        rot_deg = abs(np.degrees(t.rot))
        if rot_deg > criteria.max_rotation_deg:
            reasons.append(f"{name}: rotation {rot_deg:.3f} deg exceeds {criteria.max_rotation_deg}")
        scale_pct = max(abs(t.s1 - 1.0), abs(t.s2 - 1.0)) * 100.0
        if scale_pct > criteria.max_scale_pct:
            reasons.append(f"{name}: scale deviation {scale_pct:.3f}% exceeds {criteria.max_scale_pct}")

    score = _score(reference, transforms, (img_rows, img_cols), n_channels, criteria)
    return AcceptanceResult(passed=not reasons, score=score, reasons=reasons)


def _score(
    reference,
    transforms: "Dict[str, Transform]",
    image_shape: Tuple[float, float],
    n_channels: int,
    criteria: AcceptanceCriteria,
) -> float:
    """Combined channel-size + cross-correlation score (higher is better).

    Mirrors ``scoreChannelAlignment``: a size term (geometric-mean channel size
    normalised by image size and channel count) and an alignment term (mean
    cross-correlation peak normalised by ``cross_corr_norm``), weighted.
    """
    ref_rows, ref_cols = _channel_extent(reference.bounds)
    size_geomean = float(np.sqrt(max(ref_rows * ref_cols, 0.0)))
    img_geomean = float(np.sqrt(image_shape[0] * image_shape[1]))
    size_norm = size_geomean / (img_geomean * n_channels) if img_geomean > 0 else 0.0

    if transforms:
        mean_peak = float(np.mean([t.score for t in transforms.values()]))
        align_norm = mean_peak / criteria.cross_corr_norm if criteria.cross_corr_norm else 0.0
    else:
        align_norm = 0.0  # single channel: no alignment term

    w_size, w_cc = criteria.weight_channel_size, criteria.weight_cross_corr
    denom = w_size + w_cc
    if denom == 0:
        return float("nan")
    return (w_size * size_norm + w_cc * align_norm) / denom
