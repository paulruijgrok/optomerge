"""Optional event markers: the led_csv reader, the blue plane, and the runner flag."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from optomerge.channel import Channel
from optomerge.config import Settings
from optomerge.events import EVENT_SOURCES, led_csv_frames, load_event_frames
from optomerge.movie import RGBMovie

REPO = Path(__file__).resolve().parents[1]


def _write_led(movie: Path, times, switches, voltages=None) -> None:
    """Write the ``<stem>.csv`` / ``<stem> led.csv`` pair next to ``movie``."""
    stem = movie.with_suffix("")
    if times is not None:
        (stem.parent / f"{stem.name}.csv").write_text("".join(f"{t}\n" for t in times))
    if switches is not None:
        voltages = voltages if voltages is not None else [0.0, 1.1] * (len(switches) // 2)
        (stem.parent / f"{stem.name} led.csv").write_text(
            ",".join(f"{s:.3f}" for s in switches) + "\n"
            + ",".join(f"{v:.3f}" for v in voltages) + "\n")


# --------------------------------------------------------------------------- #
# led_csv reader
# --------------------------------------------------------------------------- #

def test_led_csv_marks_frames_inside_each_on_off_interval(tmp_path):
    movie = tmp_path / "m ROI01 01.tif"          # spaces, like the real data
    times = 1000 + 100 * np.arange(10)          # frames at 1000, 1100, ..., 1900
    # (on, off] intervals: frame at 1300 is excluded, 1600 included.
    _write_led(movie, times, [1300, 1600, 1750, 1900])
    flags, notes = led_csv_frames(movie, 10)
    assert notes == []
    assert np.flatnonzero(flags).tolist() == [4, 5, 6, 8, 9]


def test_led_csv_is_silent_without_files_and_notes_a_missing_partner(tmp_path):
    movie = tmp_path / "m.tif"
    assert led_csv_frames(movie, 5) == (None, [])
    _write_led(movie, None, [1, 2])
    flags, notes = led_csv_frames(movie, 5)
    assert flags is None and "m.csv" in notes[0]


def test_led_csv_fits_the_flags_to_the_movie_length(tmp_path):
    movie = tmp_path / "m.tif"
    _write_led(movie, [10, 20, 30, 40], [15, 40])
    short, notes = led_csv_frames(movie, 6)      # aborted timing file
    assert short.tolist() == [False, True, True, True, False, False]
    assert "left unmarked" in notes[0]
    long, notes = led_csv_frames(movie, 2)
    assert long.tolist() == [False, True] and "ignored" in notes[0]


def test_led_csv_ignores_an_unpaired_switch_and_reports_bad_files(tmp_path):
    movie = tmp_path / "m.tif"
    _write_led(movie, [10, 20, 30], [15, 25, 28], voltages=[0, 1, 0])
    flags, notes = led_csv_frames(movie, 3)
    assert flags.tolist() == [False, True, False] and "odd number" in notes[0]
    (tmp_path / "m led.csv").write_text("not,numbers\n")
    flags, notes = led_csv_frames(movie, 3)
    assert flags is None and "could not read" in notes[0]


def test_event_source_dispatch():
    assert EVENT_SOURCES[0] == "none" and "led_csv" in EVENT_SOURCES
    assert load_event_frames("none", "anything.tif", 4) == (None, [])
    with pytest.raises(ValueError):
        load_event_frames("bogus", "anything.tif", 4)


def test_events_are_off_by_default_and_configurable():
    assert Settings().events.source == "none"
    s = Settings.from_sources({"events": {"source": "led_csv", "blue_level": 1.0}})
    assert (s.events.source, s.events.blue_level) == ("led_csv", 1.0)
    assert "[events]" in s.to_toml()


# --------------------------------------------------------------------------- #
# Blue plane
# --------------------------------------------------------------------------- #

def test_flagged_frames_get_a_uniform_blue_plane():
    data = np.random.default_rng(0).random((12, 16, 4))
    ch = Channel(data, "green", np.array([[0, 11], [0, 15]]), color="green", reference=True)
    ch.vmin, ch.vmax = 0.0, 1.0
    flags = np.array([False, True, True, False])
    out = RGBMovie.from_channels([ch], {}, bg_radius=2, blue_led_frames=flags,
                                 blue_level=0.75).to_array()
    blue = out[:, :, 2, :]
    assert np.all(blue[..., flags] == 0.75) and np.all(blue[..., ~flags] == 0)
    plain = RGBMovie.from_channels([ch], {}, bg_radius=2).to_array()
    np.testing.assert_array_equal(plain[:, :, :2], out[:, :, :2])


# --------------------------------------------------------------------------- #
# End-to-end through the batch runner
# --------------------------------------------------------------------------- #

def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@pytest.mark.skipif(not (_have("scipy") and _have("skimage") and _have("tifffile")),
                    reason="requires scipy, scikit-image and tifffile")
def test_runner_marks_led_frames_only_when_enabled(tmp_path):
    import tifffile
    sys.path.insert(0, str(Path(__file__).parent))
    from test_parallel import _write_synth_movie

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    movie = in_dir / "movie 01.tif"
    _write_synth_movie(movie, n=8)
    _write_led(movie, 100 * np.arange(8), [250, 550])     # frames 3, 4, 5

    def run(out, *extra):
        proc = subprocess.run(
            [sys.executable, str(REPO / "run_optomerge.py"), "--input", str(in_dir),
             "--output", str(tmp_path / out), "--channel-order",
             "top_green_fils_bottom_red_heads", "--bunch-size", "3", *extra],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr
        (tif,) = (tmp_path / out).rglob("*_aligned.tif")
        return tifffile.imread(tif), proc.stdout

    rgb, _ = run("off")
    assert rgb.shape[0] == 8 and rgb.shape[-1] == 3
    assert not rgb[..., 2].any()

    rgb, log = run("on", "--events", "led_csv")
    per_frame = rgb[..., 2].reshape(8, -1)
    assert np.all(per_frame[[3, 4, 5]] == 127)   # 0.5 * 255, floored by the 8-bit writer
    assert not per_frame[[0, 1, 2, 6, 7]].any()
    assert "3/8 frames marked" in log
