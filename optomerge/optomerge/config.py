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


@dataclass
class ChannelSettings:
    #: Channel order / layout: "auto", "top_green_fils_bottom_red_heads", or
    #: "top_red_heads_bottom_green_fils".
    channel_order: str = "auto"
    #: Number of frames used for the detection/alignment projection (None = all).
    projection_frames: Optional[int] = None


@dataclass
class AlignmentSettings:
    #: Align on upsampled scrub images for sub-pixel accuracy.
    use_scrub: bool = False
    #: Scrub-image upscaling factor when ``use_scrub`` is set.
    upscale: int = 4
    #: Initial guesses for the scale/rotation search.
    init_rot: float = 0.0
    init_s1: float = 1.0
    init_s2: float = 1.0


@dataclass
class ProcessingSettings:
    #: Background-subtraction structuring-element radius.
    bg_radius: int = 10
    #: Frames per processing block (the FrameBunch granularity).
    bunch_size: int = 100_000


@dataclass
class RuntimeSettings:
    #: Reuse channel layout + alignment transform across the batch:
    #: "none" (independent per movie), "first" (first movie is the reference),
    #: or a path/filename to use as the reference. Intensity limits are always
    #: recomputed per movie.
    reuse_alignment: str = "none"
    verbose: bool = False
    #: List the files that would be processed, then exit without doing work.
    dry_run: bool = False


_SECTIONS = {
    "io": IOSettings,
    "channels": ChannelSettings,
    "alignment": AlignmentSettings,
    "processing": ProcessingSettings,
    "runtime": RuntimeSettings,
}


@dataclass
class Settings:
    io: IOSettings = field(default_factory=IOSettings)
    channels: ChannelSettings = field(default_factory=ChannelSettings)
    alignment: AlignmentSettings = field(default_factory=AlignmentSettings)
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
        return ChannelLayout(name=order, channel_order=order)

    def build_aligner(self):
        """Construct the :class:`Aligner` this config describes."""
        from .registration import PhaseCorrelationAligner
        a = self.alignment
        return PhaseCorrelationAligner(
            init_rot=a.init_rot, init_s1=a.init_s1, init_s2=a.init_s2,
            use_scrub=a.use_scrub, upscale=a.upscale,
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
