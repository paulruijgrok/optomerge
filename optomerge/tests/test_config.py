"""Tests for the layered configuration (mirrors FASTrack's test_config)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from optomerge.config import Settings


def _has_toml_reader() -> bool:
    """True if a TOML reader is importable (stdlib tomllib or the tomli backport)."""
    return (importlib.util.find_spec("tomllib") is not None
            or importlib.util.find_spec("tomli") is not None)


def test_layering_later_wins():
    s = Settings.from_sources(
        {"processing": {"bg_radius": 5}},
        {"processing": {"bg_radius": 20}, "channels": {"channel_order": "top_green_fils_bottom_red_heads"}},
    )
    assert s.processing.bg_radius == 20
    assert s.channels.channel_order == "top_green_fils_bottom_red_heads"
    # untouched fields keep their defaults
    assert s.processing.bunch_size == 100
    assert s.alignment.upscale == 4


def test_flat_overrides_and_none_ignored():
    s = Settings().with_overrides(bg_radius=25, use_scrub=True, projection_frames=None)
    assert s.processing.bg_radius == 25
    assert s.alignment.use_scrub is True
    # None must not clobber the default
    assert s.channels.projection_frames is None


def test_to_pipeline_kwargs_maps_sections():
    k = Settings().with_overrides(bg_radius=7, bunch_size=500, projection_frames=50).to_pipeline_kwargs()
    assert k["bg_radius"] == 7
    assert k["bunch_size"] == 500
    assert k["projection_frames"] == 50
    assert k["verbose"] is False
    # adapters build the right domain objects
    from optomerge import ChannelLayout, PhaseCorrelationAligner
    assert isinstance(k["layout"], ChannelLayout)
    assert isinstance(k["aligner"], PhaseCorrelationAligner)


def test_build_layout_and_aligner_reflect_config():
    s = Settings.from_sources(
        {"channels": {"channel_order": "top_red_heads_bottom_green_fils"},
         "alignment": {"use_scrub": True, "upscale": 8}},
    )
    layout = s.build_layout()
    aligner = s.build_aligner()
    assert layout.channel_order == "top_red_heads_bottom_green_fils"
    assert aligner.use_scrub is True and aligner.upscale == 8


def test_unknown_section_and_key_raise():
    for bad in ({"bogus": {}}, {"processing": {"nope": 1}}):
        with pytest.raises(KeyError):
            Settings.from_sources(bad)


def test_with_overrides_unknown_field_raises():
    with pytest.raises(KeyError):
        Settings().with_overrides(not_a_field=1)


@pytest.mark.skipif(not _has_toml_reader(), reason="no TOML reader (install tomli on Python < 3.11)")
def test_toml_roundtrip(tmp_path):
    original = Settings.from_sources(
        {"io": {"suffix": "_reg", "overwrite": True},
         "channels": {"channel_order": "top_green_fils_bottom_red_heads", "projection_frames": 120},
         "alignment": {"use_scrub": True, "init_s1": 1.02},
         "processing": {"bg_radius": 12}},
    )
    p = tmp_path / "run_config.toml"
    p.write_text(original.to_toml(), encoding="utf-8")
    reloaded = Settings.from_toml(str(p))
    assert reloaded.as_dict() == original.as_dict()


@pytest.mark.skipif(not _has_toml_reader(), reason="no TOML reader (install tomli on Python < 3.11)")
def test_shipped_example_toml_loads():
    example = Path(__file__).resolve().parents[2] / "optomerge.example.toml"
    if not example.exists():
        pytest.skip("optomerge.example.toml not found")
    s = Settings.from_toml(str(example))
    assert s.io.input == "test_data"
    assert s.channels.channel_order == "auto"


@pytest.mark.parametrize("module_name", ["run_optomerge", "test_pipeline", "profile_pipeline"])
def test_cli_field_map_targets_valid_fields(module_name):
    # Every CLI dest in each script must map to a real Settings field, and
    # applying them all must not raise (catches typos / renamed fields).
    mod = _load_script(module_name)
    _bool_fields = ("overwrite", "use_scrub", "verbose", "dry_run")
    sample = {f: (True if f in _bool_fields else 1) for f in mod._CLI_TO_FIELD.values()}
    s = Settings().with_overrides(**sample)  # raises KeyError on any bad field
    assert s is not None


def _load_script(name: str):
    """Import a top-level entry-point script by name (repo root on sys.path)."""
    import importlib
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(name)
