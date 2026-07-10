"""Layered configuration for an optomerge alignment run.

Settings are grouped by concern -- ``io`` (where movies are read from / written
to), ``channels`` (channel detection), ``alignment`` (registration strategy),
``processing`` (background subtraction / chunking), and ``runtime`` (execution)
-- so a setup can be composed from small, independent pieces.

The :meth:`Settings.from_sources` constructor is the seam for that composition:
it merges any number of layers (built-in defaults -> config file(s) -> CLI
overrides), each later layer overriding the earlier ones. Layers are plain
nested dicts; loading them from TOML is a thin adapter on top
(:meth:`Settings.from_toml`), and :meth:`Settings.to_toml` writes the fully
resolved config back out so every run's exact parameters are recorded alongside
its output.

This mirrors the layered ``Settings`` pattern used in the FASTrack package.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class IOSettings:
    #: Root directory of raw movies (scanned recursively for .tif/.tiff).
    input: str = "test_data"
    #: Root directory for aligned output; input folder structure is mirrored.
    output: str = "output_temp"
    #: Suffix inserted before the extension of each output filename.
    suffix: str = "_aligned"
    #: Overwrite existing output files instead of skipping them.
    overwrite: bool = False
    #: RGB output bit depth: 8 (default; compact, fixed-range display, like the
    #: MATLAB reference) or 16 (more dynamic range, auto-contrasted in ImageJ).
    rgb_bitdepth: int = 8


@dataclass
class ChannelSettings:
    #: Channel order / layout: "auto", "top_green_fils_bottom_red_heads", or
    #: "top_red_heads_bottom_green_fils".
    channel_order: str = "auto"
    #: Number of frames used for the detection/alignment projection (None = all).
    projection_frames: Optional[int] = None
    #: Channel-finding method: "row_profile" (default, robust rectangles) or
    #: "line_search" (legacy k-means + slanted-line search).
    segmentation: str = "row_profile"


@dataclass
class AlignmentSettings:
    #: Registration algorithm: "phase" (default; FFT phase correlation, general
    #: purpose) or "feature" (head-to-filament distance minimisation, tailored to
    #: point-vs-line OptoSplit channels -- see FeatureDistanceAligner).
    method: str = "phase"
    #: Sampled raw frames used by the "feature" aligner (spread across the movie).
    feature_frames: int = 60
    #: "feature" head threshold, in std-devs above the moving channel's per-frame mean.
    head_sigma: float = 3.0
    #: "feature" filament threshold, in std-devs above the reference channel's mean.
    filament_sigma: float = 2.0
    #: "feature" head blob area gate (pixels): keep blobs within [min, max]; the
    #: upper bound rejects filament crossings / bleed that aren't point-like heads.
    head_min_area: int = 4
    head_max_area: int = 60
    #: "feature" filament despeckle: drop mask components smaller than this (pixels)
    #: so background noise is not treated as filament.
    filament_min_area: int = 20
    #: "feature" robustness cap (pixels): a head farther than this from any filament
    #: is treated as an orphan (red-only object / undetected filament) and cannot
    #: bias the fit. Also the inlier threshold for the reported score.
    distance_cap: float = 6.0
    #: "feature" down-weight stuck objects: weight each head by 1/persistence so a
    #: long-lived (stuck) object counts once, not once-per-frame. Restores field
    #: coverage when a few stuck objects would otherwise dominate the fit.
    deweight_stuck: bool = False
    #: "feature" radius (pixels) within which detections in different frames are
    #: treated as the same stuck object for the persistence count.
    stuck_radius: float = 2.0
    #: "feature" also fit a small bounded rotation between the channels, layered
    #: on the translation fit (for a field-dependent residual = optical rotation).
    fit_rotation: bool = False
    #: "feature" also fit a small bounded isotropic scale between the channels.
    fit_scale: bool = False
    #: "feature" rotation search bound (degrees): |rotation| <= this. Kept tight
    #: so the extra degree of freedom cannot over-fit.
    feature_rotation_max_deg: float = 2.0
    #: "feature" isotropic scale search bound (percent): |scale - 1| * 100 <= this.
    feature_scale_max_pct: float = 2.0
    #: Align on upsampled scrub images for sub-pixel accuracy.
    use_scrub: bool = False
    #: Scrub-image upscaling factor when ``use_scrub`` is set.
    upscale: int = 4
    #: Initial guesses for the scale/rotation search.
    init_rot: float = 0.0
    init_s1: float = 1.0
    init_s2: float = 1.0
    #: If > 0, constrain the fitted translation to ±max_shift px of the origin,
    #: ignoring spurious far-off correlation peaks. 0 = unconstrained.
    max_shift: float = 0.0
    #: Search for inter-channel rotation + scale (True) or fit translation only
    #: (False). For fixed optics (OptoSplit) a free rotation/scale search tends
    #: to over-fit; translation-only is often more stable.
    fit_scale_rotation: bool = True


@dataclass
class ProcessingSettings:
    #: Background-subtraction structuring-element radius.
    bg_radius: int = 10
    #: Frames held in memory / processed per block (the FrameBunch granularity).
    #: Small keeps memory bounded; ~100 mirrors the MATLAB reference.
    bunch_size: int = 100
    #: Projection used for per-channel normalisation limits: "max" (default;
    #: tracks per-frame peaks so moving filaments don't saturate) or "mean".
    #: The projection is median-filtered first to remove cosmic rays.
    norm_projection: str = "max"
    #: Extra percentile fraction excluded at each end when setting the limits.
    #: 0 = plain min/max after median-despeckle (keeps sparse features like heads);
    #: raise only if a channel still has outliers the median missed.
    norm_exclude: float = 0.0


@dataclass
class CalibrationSettings:
    #: Channel/alignment finding strategy: "single" (one projection of the whole
    #: movie) or "robust" (best-of-N over frame chunks, acceptance-filtered).
    mode: str = "single"
    #: Frames per projection block for robust mode (projectionChunkSize).
    chunk_size: int = 400
    #: Passing candidates required before picking the best (minNumTopAlignments).
    min_candidates: int = 10
    #: Max blocks tried before giving up on a movie (maxNumTrials).
    max_trials: int = 30


@dataclass
class AcceptanceSettings:
    #: Minimum reference-channel row extent as a fraction of image height,
    #: times the number of channels (stacked channels each span ~1/N).
    min_size_row_frac: float = 0.7
    #: Minimum reference-channel column extent as a fraction of image width.
    min_size_col_frac: float = 0.85
    #: Maximum allowed |t1|/|t2| translation between channels (pixels).
    max_translation_px: float = 25.0
    #: Maximum allowed |rotation| between channels (degrees).
    max_rotation_deg: float = 2.0
    #: Maximum allowed scale deviation |s - 1| * 100 (percent).
    max_scale_pct: float = 2.0
    #: Scoring weights (channel size vs cross-correlation peak).
    size_weight: float = 1.0
    cross_corr_weight: float = 3.0
    #: Normalisation for the cross-correlation peak.
    cross_corr_norm: float = 0.1


@dataclass
class GroupingSettings:
    #: How to partition movies into sets that share one alignment (used when
    #: runtime.reuse_alignment != "none"): "run" (whole batch is one set),
    #: "token" (group by a regex marker in the filename), or "folder".
    by: str = "run"
    #: Regex marker for the "token" strategy (channel/date marker).
    token_pattern: str = r"_ch\d\d_"


@dataclass
class RuntimeSettings:
    #: Reuse channel layout + alignment transform across each set (see
    #: [grouping]): "none" (independent per movie), "first" (each set's first
    #: movie is its reference), or a path/filename to use as a predetermined
    #: reference. Intensity limits are always recomputed per movie.
    reuse_alignment: str = "none"
    verbose: bool = False
    #: List the files that would be processed, then exit without doing work.
    dry_run: bool = False


_SECTIONS = {
    "io": IOSettings,
    "channels": ChannelSettings,
    "alignment": AlignmentSettings,
    "calibration": CalibrationSettings,
    "acceptance": AcceptanceSettings,
    "grouping": GroupingSettings,
    "processing": ProcessingSettings,
    "runtime": RuntimeSettings,
}


@dataclass
class Settings:
    io: IOSettings = field(default_factory=IOSettings)
    channels: ChannelSettings = field(default_factory=ChannelSettings)
    alignment: AlignmentSettings = field(default_factory=AlignmentSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    acceptance: AcceptanceSettings = field(default_factory=AcceptanceSettings)
    grouping: GroupingSettings = field(default_factory=GroupingSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    # ------------------------------------------------------------------ #
    # Construction / layering
    # ------------------------------------------------------------------ #
    @classmethod
    def from_sources(cls, *layers: Mapping[str, Mapping[str, Any]]) -> "Settings":
        """Build settings by merging nested-dict ``layers`` left-to-right.

        Each layer looks like ``{"io": {...}, "alignment": {...}}`` and may set
        only the keys it cares about; later layers win. Unknown sections/keys
        raise, to catch typos in config files early.
        """
        merged: Dict[str, Dict[str, Any]] = {name: {} for name in _SECTIONS}
        for layer in layers:
            if not layer:
                continue
            for section, values in layer.items():
                if section not in _SECTIONS:
                    raise KeyError("Unknown settings section: %r" % section)
                valid = {f.name for f in fields(_SECTIONS[section])}
                for key, value in dict(values).items():
                    if key not in valid:
                        raise KeyError("Unknown %s setting: %r" % (section, key))
                    merged[section][key] = value
        return cls(**{name: _SECTIONS[name](**vals) for name, vals in merged.items()})

    def with_overrides(self, **flat: Any) -> "Settings":
        """Return a copy with flat ``field=value`` overrides.

        Field names are unique across sections, so callers (e.g. the CLI) can
        pass them flat without knowing which section they live in. ``None``
        values are ignored, so unset CLI flags don't clobber defaults.
        """
        index = {}
        for name, sect_cls in _SECTIONS.items():
            for f in fields(sect_cls):
                index[f.name] = name
        updates: Dict[str, Dict[str, Any]] = {name: {} for name in _SECTIONS}
        for key, value in flat.items():
            if value is None:
                continue
            if key not in index:
                raise KeyError("Unknown setting: %r" % key)
            updates[index[key]][key] = value
        new_sections = {
            name: replace(getattr(self, name), **updates[name]) for name in _SECTIONS
        }
        return replace(self, **new_sections)

    @classmethod
    def from_toml(cls, *paths: str) -> "Settings":
        """Load and merge TOML config files.

        Uses the standard-library ``tomllib`` on Python 3.11+ and falls back to
        the ``tomli`` backport on 3.9/3.10 (installed automatically via the
        ``python_version < '3.11'`` dependency marker).
        """
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover - py<3.11
            try:
                import tomli as tomllib  # backport for 3.9/3.10
            except ModuleNotFoundError:
                raise RuntimeError(
                    "Reading TOML config on Python < 3.11 needs 'tomli'. "
                    "Install it with:  pip install tomli"
                )
        layers = []
        for path in paths:
            with open(path, "rb") as f:
                layers.append(tomllib.load(f))
        return cls.from_sources(*layers)

    # ------------------------------------------------------------------ #
    # Adapters
    # ------------------------------------------------------------------ #
    def build_layout(self):
        """Construct the :class:`ChannelLayout` this config describes."""
        from .layout import ChannelLayout
        order = self.channels.channel_order
        return ChannelLayout(name=order, channel_order=order,
                             segmentation_method=self.channels.segmentation)

    def build_aligner(self):
        """Construct the :class:`Aligner` this config describes."""
        a = self.alignment
        if a.method == "feature":
            from .feature_registration import FeatureDistanceAligner
            # The feature aligner needs a bounded search; default to 30 px when
            # max_shift is left at its "unconstrained" sentinel of 0.
            return FeatureDistanceAligner(
                max_shift=a.max_shift if a.max_shift > 0 else 30.0,
                head_sigma=a.head_sigma, filament_sigma=a.filament_sigma,
                min_head_area=a.head_min_area, max_head_area=a.head_max_area,
                filament_min_area=a.filament_min_area, distance_cap=a.distance_cap,
                deweight_stuck=a.deweight_stuck, stuck_radius=a.stuck_radius,
                fit_rotation=a.fit_rotation, fit_scale=a.fit_scale,
                rotation_max_deg=a.feature_rotation_max_deg,
                scale_max_pct=a.feature_scale_max_pct,
            )
        from .registration import PhaseCorrelationAligner
        return PhaseCorrelationAligner(
            init_rot=a.init_rot, init_s1=a.init_s1, init_s2=a.init_s2,
            use_scrub=a.use_scrub, upscale=a.upscale, max_shift=a.max_shift,
            fit_scale_rotation=a.fit_scale_rotation,
        )

    def build_acceptance(self):
        """Construct the :class:`AcceptanceCriteria` this config describes."""
        from .acceptance import AcceptanceCriteria
        a = self.acceptance
        return AcceptanceCriteria(
            min_size_frac=(a.min_size_row_frac, a.min_size_col_frac),
            max_translation_px=a.max_translation_px,
            max_rotation_deg=a.max_rotation_deg,
            max_scale_pct=a.max_scale_pct,
            weight_channel_size=a.size_weight,
            weight_cross_corr=a.cross_corr_weight,
            cross_corr_norm=a.cross_corr_norm,
        )

    def build_calibrator(self):
        """Construct the :class:`Calibrator` this config describes.

        ``[calibration].mode = "robust"`` selects the best-of-N
        :class:`~optomerge.calibration.RobustCalibrator`; anything else selects
        the :class:`~optomerge.calibration.SingleProjectionCalibrator`.
        """
        from .calibration import RobustCalibrator, SingleProjectionCalibrator
        layout = self.build_layout()
        aligner = self.build_aligner()
        criteria = self.build_acceptance()
        c = self.calibration
        norm_from_max = self.processing.norm_projection == "max"
        norm_exclude = self.processing.norm_exclude
        if c.mode == "robust":
            return RobustCalibrator(
                layout=layout, aligner=aligner, criteria=criteria,
                chunk_size=c.chunk_size, min_candidates=c.min_candidates,
                max_trials=c.max_trials, verbose=self.runtime.verbose,
                norm_from_max=norm_from_max, norm_exclude=norm_exclude,
            )
        return SingleProjectionCalibrator(
            layout=layout, aligner=aligner,
            projection_frames=self.channels.projection_frames,
            criteria=criteria, verbose=self.runtime.verbose,
            norm_from_max=norm_from_max, norm_exclude=norm_exclude,
            feature_frames=self.alignment.feature_frames,
        )

    def to_pipeline_kwargs(self) -> Dict[str, Any]:
        """Flatten to the keyword arguments accepted by ``MergePipeline``.

        Usage: ``MergePipeline(source, **settings.to_pipeline_kwargs())``.
        """
        return {
            "layout": self.build_layout(),
            "aligner": self.build_aligner(),
            "bunch_size": self.processing.bunch_size,
            "bg_radius": self.processing.bg_radius,
            "projection_frames": self.channels.projection_frames,
            "verbose": self.runtime.verbose,
        }

    def as_dict(self) -> Dict[str, Any]:
        """Nested-dict view (round-trips through ``from_sources``)."""
        return asdict(self)

    def to_toml(self) -> str:
        """Serialise the fully-resolved config to a TOML string.

        Written next to a run's output as a provenance record; the result loads
        back through :meth:`from_toml`. Fields that are ``None`` (no TOML null)
        are emitted as commented-out lines so the resolved value is still visible
        without breaking a round-trip.
        """
        lines: List[str] = [
            "# optomerge resolved run configuration.",
            "# Load with: Settings.from_toml('run_config.toml')",
            "",
        ]
        for section in _SECTIONS:
            lines.append(f"[{section}]")
            for f in fields(_SECTIONS[section]):
                value = getattr(getattr(self, section), f.name)
                if value is None:
                    lines.append(f"# {f.name} =            # unset (all / default)")
                else:
                    lines.append(f"{f.name} = {_toml_value(value)}")
            lines.append("")
        return "\n".join(lines)


def _toml_value(value: Any) -> str:
    """Render a scalar/list value as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"Cannot serialise {type(value).__name__} to TOML: {value!r}")
