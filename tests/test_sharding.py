"""Tests for splitting a batch across cluster jobs (--num-shards / --shard-index)
and for sizing thread pools to the CPU allocation."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from optomerge import available_cpus, group_movies, select_shard

REPO = Path(__file__).resolve().parents[1]


def _paths(n, folder="d"):
    return [Path(f"{folder}/m{i:02d}.tif") for i in range(n)]


def _flat(groups):
    return [p for _, ms in groups for p in ms]


@pytest.mark.parametrize("n_files,n_shards", [(10, 3), (7, 7), (3, 5), (31, 4)])
def test_shards_partition_files_exactly_once(n_files, n_shards):
    groups = group_movies(_paths(n_files), by="run")
    seen = []
    for i in range(n_shards):
        seen += _flat(select_shard(groups, n_shards, i, keep_groups_together=False))
    assert sorted(seen) == sorted(_paths(n_files))   # complete, no duplicates
    assert len(seen) == n_files


def test_independent_files_are_balanced():
    groups = group_movies(_paths(10), by="run")
    sizes = [len(_flat(select_shard(groups, 3, i, keep_groups_together=False)))
             for i in range(3)]
    assert sorted(sizes) == [3, 3, 4]


def test_dependent_sets_stay_together():
    files = _paths(4, "a") + _paths(3, "b") + _paths(2, "c")
    groups = group_movies(files, by="folder")
    shards = [select_shard(groups, 2, i, keep_groups_together=True) for i in range(2)]
    # every set lands whole in exactly one shard
    for key, members in groups:
        owners = [i for i, sh in enumerate(shards) if (key, members) in sh]
        assert len(owners) == 1
    assert sorted(_flat(shards[0]) + _flat(shards[1])) == sorted(files)


def test_one_shard_is_identity_and_bad_index_raises():
    groups = group_movies(_paths(5), by="run")
    assert select_shard(groups, 1, 0, keep_groups_together=False) == groups
    with pytest.raises(ValueError):
        select_shard(groups, 3, 3, keep_groups_together=False)
    with pytest.raises(ValueError):
        select_shard(groups, 0, 0, keep_groups_together=False)


def test_more_shards_than_files_gives_empty_shards():
    groups = group_movies(_paths(2), by="run")
    assert select_shard(groups, 5, 4, keep_groups_together=False) == []


def test_available_cpus_respects_slurm_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert available_cpus() == 1
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    assert available_cpus() >= 1
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert 1 <= available_cpus() <= (os.cpu_count() or 1)


def test_cli_dry_run_lists_only_this_shard(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    for i in range(5):
        (src / f"m{i}.tif").write_bytes(b"")
    listed = []
    for i in range(2):
        out = subprocess.run(
            [sys.executable, str(REPO / "run_optomerge.py"), "--input", str(src),
             "--output", str(tmp_path / "out"), "--dry-run",
             "--num-shards", "2", "--shard-index", str(i)],
            capture_output=True, text=True, check=True).stdout
        assert f"shard {i}/2" in out
        listed += [ln.strip() for ln in out.splitlines() if ln.strip().endswith(".tif")
                   and "->" not in ln]
    assert sorted(listed) == [f"m{i}.tif" for i in range(5)]


def test_sharded_run_matches_single_run_and_resumes(tmp_path):
    """Two shards writing into one output dir == one unsharded run, byte for
    byte; re-running a shard skips finished movies (requeue-safe)."""
    pytest.importorskip("skimage")
    tifffile = pytest.importorskip("tifffile")
    import numpy as np
    from test_parallel import _write_synth_movie

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(3):
        _write_synth_movie(in_dir / f"movie_{i:02d}.tif", shift=1 + i, seed=i)

    def run(out, *extra):
        return subprocess.run(
            [sys.executable, str(REPO / "run_optomerge.py"),
             "--input", str(in_dir), "--output", str(out),
             "--channel-order", "top_green_fils_bottom_red_heads", *extra],
            capture_output=True, text=True, timeout=300)

    single = run(tmp_path / "single")
    assert single.returncode == 0, single.stderr
    for i in range(2):
        r = run(tmp_path / "sharded", "--num-shards", "2", "--shard-index", str(i))
        assert r.returncode == 0, r.stderr

    a = sorted((tmp_path / "single").rglob("*_aligned.tif"))
    b = sorted((tmp_path / "sharded").rglob("*_aligned.tif"))
    assert [p.name for p in a] == [p.name for p in b] and len(a) == 3
    for x, y in zip(a, b):
        np.testing.assert_array_equal(tifffile.imread(x), tifffile.imread(y))
    logs = sorted(p.name for p in (tmp_path / "sharded").glob("run_log*.txt"))
    assert logs == ["run_log_shard000of002.txt", "run_log_shard001of002.txt"]
    assert not list((tmp_path / "sharded").rglob("*.partial"))

    again = run(tmp_path / "sharded", "--num-shards", "2", "--shard-index", "0")
    assert again.returncode == 0
    log0 = (tmp_path / "sharded" / "run_log_shard000of002.txt").read_text()
    assert "[SKIP] output exists" in log0
