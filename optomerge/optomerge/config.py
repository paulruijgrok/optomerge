"""
optomerge.config
================
Pydantic-based configuration models for the optomerge pipeline.

Configuration is loaded from a TOML file and validated at startup.  All
parameters have sensible defaults so the config file is entirely optional.

Resolution order (last wins):
    built-in defaults → TOML config file → CLI flags

Config file search order (first found wins):
    1. Path supplied via ``--config FILE``
    2. ``./optomerge.toml``  (next to the script being run)
    3. ``~/.config/optomerge/optomerge.toml``  (user-global defaults)

Python ≥ 3.11 uses the stdlib ``tomllib``; earlier versions require the
``tomli`` back-port (listed as an optional dependency in ``pyproject.toml``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# tomllib compatibility (3.11+ stdlib; tomli back-port for 3.9 / 3.10)
# ---------------------------------------------------------------------------
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib          # type: ignore[no-reuse-declaration]
    except ImportError as _err:
        raise ImportError(
            "optomerge requires 'tomli' on Python < 3.11.  "
            "Install it with:  pip install tomli"
        ) from _err


# ---------------------------------------------------------------------------
# Sub-section models
# ---------------------------------------------------------------------------

class IoConfig(BaseModel):
    """File I/O settings."""

    input: str = Field(
        "test_data",
        description="Root directory that contains the raw TIFF movies.",
    )
    output: str = Field(
        "output_temp",
        description="Root directory where aligned output files are written.",
    )
    suffix: str = Field(
        "_aligned",
        description="String appended to each output filename before the extension.",
    )
    format: Literal["rgb_tiff", "ome_tiff", "zarr", "split_channels"] = Field(
        "rgb_tiff",
        description=(
            "Output file format.  Currently only 'rgb_tiff' is fully "
            "implemented; the others are reserved for future use."
        ),
    )
    overwrite: bool = Field(
        False,
        description=(
            "If False (default), skip files whose output path already exists. "
            "If True, overwrite existing outputs."
        ),
    )


class ProjectionConfig(BaseModel):
    """Settings for the mean-projection used in channel / alignment detection."""

    frames: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Maximum number of frames to include in the projection.  "
            "null (default) uses all frames."
        ),
    )
    bg_radius: int = Field(
        10,
        ge=1,
        description=(
            "Radius (pixels) of the structuring element used for morphological-"
            "opening background subtraction, or the sigma for Gaussian "
            "background subtraction."
        ),
    )
    bg_method: Literal["morph_open", "gaussian"] = Field(
        "morph_open",
        description=(
            "'morph_open' (default): morphological opening — faithful "
            "reproduction of the MATLAB behaviour.  "
            "'gaussian': Gaussian blur — ~10–30× faster, similar quality."
        ),
    )
    channel_order: Literal[
        "auto",
        "top_green_fils_bottom_red_heads",
        "top_red_heads_bottom_green_fils",
    ] = Field(
        "auto",
        description=(
            "Spatial order of the two imaging channels on the detector.  "
            "'auto' (default) detects the order from the image intensity.  "
            "Override when auto-detection fails."
        ),
    )


class AutoMethodThresholds(BaseModel):
    """Thresholds used when align_method = 'auto'.

    When the computed alignment shows rotation **and** scale deviations below
    these limits, the faster ``fft_shift`` method is selected automatically;
    otherwise the full ``affine`` transform is used.
    """

    rotation_deg: float = Field(
        0.5,
        gt=0,
        description=(
            "Maximum absolute rotation (degrees) that is considered "
            "'translation-only' for the purpose of auto-selecting fft_shift. "
            "Default 0.5°."
        ),
    )
    scale_percent: float = Field(
        0.5,
        gt=0,
        description=(
            "Maximum scale deviation (percent, i.e. |s−1|×100) that is "
            "considered 'translation-only'.  Default 0.5 %."
        ),
    )


class AlignmentConfig(BaseModel):
    """Parameters controlling how the inter-channel alignment is computed."""

    mode: Literal["robust", "single"] = Field(
        "robust",
        description=(
            "'robust' (default): project multiple frame chunks, score each "
            "candidate, pick the best — equivalent to MATLAB "
            "findChannelsAndAlignment.  "
            "'single': one projection of all (or projection.frames) frames."
        ),
    )
    reuse: str = Field(
        "none",
        description=(
            "Batch-level alignment reuse.  "
            "'none' (default): every movie is aligned independently.  "
            "'first': use the first movie in the batch as the geometric "
            "reference; subsequent movies skip alignment computation.  "
            "Any other value is interpreted as a file path pointing to the "
            "reference movie."
        ),
    )
    robust_chunk_size: int = Field(
        400,
        ge=10,
        description="Frames per projection chunk in 'robust' mode.  Default 400.",
    )
    robust_min_candidates: int = Field(
        10,
        ge=1,
        description=(
            "Minimum number of valid chunk candidates to collect before "
            "selecting the best.  Default 10."
        ),
    )
    robust_max_trials: int = Field(
        30,
        ge=1,
        description=(
            "Hard upper limit on the number of chunks tried before giving up "
            "and using the best candidate found so far.  Default 30."
        ),
    )
    auto_method_thresholds: AutoMethodThresholds = Field(
        default_factory=AutoMethodThresholds,
        description=(
            "Thresholds for automatic selection of the frame-alignment method "
            "when frames.align_method = 'auto'."
        ),
    )

    @model_validator(mode="after")
    def _check_trials_ge_candidates(self) -> "AlignmentConfig":
        if self.robust_max_trials < self.robust_min_candidates:
            raise ValueError(
                f"robust_max_trials ({self.robust_max_trials}) must be >= "
                f"robust_min_candidates ({self.robust_min_candidates})"
            )
        return self


class FramesConfig(BaseModel):
    """Settings applied when transforming individual frames."""

    align_method: Literal["auto", "affine", "fft_shift", "quick_and_dirty"] = Field(
        "auto",
        description=(
            "Frame-alignment method applied after computing alignment parameters.\n"
            "  'auto'            — use fft_shift when rotation < threshold AND "
            "scale deviation < threshold (see alignment.auto_method_thresholds); "
            "otherwise affine.\n"
            "  'affine'          — full affine transform (most accurate, slowest).\n"
            "  'fft_shift'       — sub-pixel Fourier shift (~2–3× faster; "
            "ignores residual rotation/scale).\n"
            "  'quick_and_dirty' — integer-pixel shift, no interpolation "
            "(~20–50× faster, ±0.5 px)."
        ),
    )


class ComputeConfig(BaseModel):
    """Parallelism and backend settings."""

    backend: Literal["auto", "cv2", "scipy"] = Field(
        "auto",
        description=(
            "Computational backend for background subtraction and affine "
            "transform.  'auto' (default) uses OpenCV when available."
        ),
    )
    n_workers: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Number of CPU threads for frame-level parallel processing.  "
            "null (default) uses all logical CPUs.  Use 1 to disable "
            "parallelism."
        ),
    )
    parallel_files: bool = Field(
        False,
        description=(
            "Process multiple movies simultaneously using separate processes.  "
            "When enabled, frame-level workers is forced to 1 to avoid nested "
            "parallelism.  Best for batches of many short movies."
        ),
    )


class LoggingConfig(BaseModel):
    """Terminal output and log-file settings."""

    verbose: bool = Field(
        False,
        description="Print detailed per-step alignment parameters.",
    )
    show_progress: bool = Field(
        True,
        description=(
            "Show live progress bars for slow steps.  Set False when piping "
            "terminal output to a file."
        ),
    )
    dry_run: bool = Field(
        False,
        description="List files that would be processed without doing any work.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO",
        description="Verbosity of the log file written to the output directory.",
    )


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------

class OptomergeConfig(BaseModel):
    """Root configuration for the optomerge batch pipeline.

    All fields have defaults, so an empty config file (or no config file at
    all) is perfectly valid.

    Example
    -------
    Load from a TOML file::

        cfg = OptomergeConfig.from_toml(Path("optomerge.toml"))

    Auto-discover and load (falls back to defaults if no file found)::

        cfg = OptomergeConfig.find_and_load()

    Override a single value programmatically::

        cfg.alignment.mode = "single"
    """

    io:         IoConfig         = Field(default_factory=IoConfig)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    alignment:  AlignmentConfig  = Field(default_factory=AlignmentConfig)
    frames:     FramesConfig     = Field(default_factory=FramesConfig)
    compute:    ComputeConfig    = Field(default_factory=ComputeConfig)
    logging:    LoggingConfig    = Field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: Path) -> "OptomergeConfig":
        """Load and validate a config from a TOML file at *path*.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        pydantic.ValidationError
            If any value fails validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        return cls.model_validate(raw)

    @classmethod
    def find_and_load(
        cls,
        explicit_path: Optional[Path] = None,
        search_cwd: bool = True,
    ) -> "OptomergeConfig":
        """Search for a config file and return a validated config.

        Search order (first found wins):
            1. *explicit_path* (e.g. from ``--config FILE``)
            2. ``./optomerge.toml``  (when *search_cwd* is True)
            3. ``~/.config/optomerge/optomerge.toml``

        If no file is found, returns an instance with all default values.

        Parameters
        ----------
        explicit_path : Path or None
            Config file path supplied explicitly (e.g. via CLI).
        search_cwd : bool
            Whether to look for ``optomerge.toml`` in the current working
            directory.  Default True.

        Returns
        -------
        OptomergeConfig
        """
        candidates: list[Path] = []
        if explicit_path is not None:
            candidates.append(Path(explicit_path))
        if search_cwd:
            candidates.append(Path.cwd() / "optomerge.toml")
        candidates.append(
            Path.home() / ".config" / "optomerge" / "optomerge.toml"
        )

        for path in candidates:
            if path.exists():
                cfg = cls.from_toml(path)
                cfg._source_path = path        # type: ignore[attr-defined]
                return cfg

        return cls()                           # pure defaults, no file

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def source_path(self) -> Optional[Path]:
        """Return the path of the config file that was loaded, or None."""
        return getattr(self, "_source_path", None)

    def to_toml_string(self) -> str:
        """Serialise this config back to a TOML string.

        Useful for logging the resolved configuration alongside a run's
        output, or for generating a starting-point config file.
        """
        lines: list[str] = []

        def _val(v: object) -> str:
            if v is None:
                return "null   # uncomment and set a value to override"
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, str):
                return f'"{v}"'
            return str(v)

        def _section(name: str, model: BaseModel) -> None:
            lines.append(f"\n[{name}]")
            for fname, finfo in model.model_fields.items():
                val = getattr(model, fname)
                if isinstance(val, BaseModel):
                    # Nested model — rendered as its own sub-section
                    lines.append(f"\n[{name}.{fname}]")
                    for sfname, sfinfo in val.model_fields.items():
                        sval = getattr(val, sfname)
                        desc = sfinfo.description or ""
                        short = desc.split(".")[0].split("\n")[0]
                        if short:
                            lines.append(f"# {short}")
                        lines.append(f"{sfname} = {_val(sval)}")
                    continue
                desc = finfo.description or ""
                short = desc.split(".")[0].split("\n")[0]
                if short:
                    lines.append(f"# {short}")
                lines.append(f"{fname} = {_val(val)}")

        for section_name, section_model in [
            ("io",         self.io),
            ("projection", self.projection),
            ("alignment",  self.alignment),
            ("frames",     self.frames),
            ("compute",    self.compute),
            ("logging",    self.logging),
        ]:
            _section(section_name, section_model)

        return "\n".join(lines).lstrip()
