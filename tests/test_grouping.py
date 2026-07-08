"""Tests for movie-set grouping (pure; no heavy deps)."""
from __future__ import annotations

from pathlib import Path

import pytest

from optomerge import group_movies, grouping_modes
from optomerge.grouping import DEFAULT_TOKEN_PATTERN

PATHS = [
    Path("data/A/20180620_ch03_prep_movie 01.tif"),
    Path("data/A/20180620_ch03_prep_movie 02.tif"),
    Path("data/A/20180620_ch12_other_movie 01.tif"),
    Path("data/B/20170803_ch12_thing_movie 01.tif"),
    Path("data/B/no_marker_movie.tif"),
]


def test_run_mode_is_one_set():
    groups = group_movies(PATHS, by="run")
    assert len(groups) == 1
    _key, members = groups[0]
    assert len(members) == len(PATHS)


def test_token_mode_groups_by_marker():
    groups = dict(group_movies(PATHS, by="token", token_pattern=DEFAULT_TOKEN_PATTERN))
    # ch03 -> 2 movies, ch12 -> 2 movies (across folders), no-marker -> its own set
    assert len(groups["_ch03_"]) == 2
    assert len(groups["_ch12_"]) == 2
    assert any(k.startswith("__notoken__") for k in groups)


def test_token_mode_preserves_order():
    groups = group_movies(PATHS, by="token")
    keys = [k for k, _ in groups]
    assert keys[0] == "_ch03_"          # first-seen key first
    assert keys[1] == "_ch12_"


def test_folder_mode_groups_by_directory():
    groups = dict(group_movies(PATHS, by="folder"))
    assert set(groups) == {"data/A", "data/B"}
    assert len(groups["data/A"]) == 3
    assert len(groups["data/B"]) == 2


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        group_movies(PATHS, by="nope")


def test_available_modes_registered():
    assert set(grouping_modes()) == {"run", "token", "folder"}


def test_grouping_config_defaults():
    from optomerge import Settings
    g = Settings().grouping
    assert g.by == "run"
    assert g.token_pattern == DEFAULT_TOKEN_PATTERN
