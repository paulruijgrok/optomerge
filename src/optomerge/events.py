"""
Optional per-frame event markers, shown in the output's blue plane.

Some experiments switch an external perturbation (e.g. a blue LED) on and off
during acquisition. The merge can paint the blue plane of every frame recorded
while it was on, so the perturbation is visible at a glance in the RGB movie
(see ``RGBMovie.from_channels(blue_led_frames=...)``).

How the on/off times are stored is specific to each setup, so every format is a
named *source* below, and none is active by default (``[events] source =
"none"``). A source maps a raw movie to a boolean array with one entry per frame,
plus QC notes; add a format by writing a reader and registering it in
``_SOURCES``.

``led_csv``
    The format of the original MATLAB pipeline (``getBlueLEDtimings`` in
    ``alignRGB.m``). Next to ``<stem>.tif`` sit two files:

    * ``<stem>.csv``: one acquisition timestamp per frame, one per line.
    * ``<stem> led.csv``: the first row holds switch times on the same clock,
      in on/off pairs; the second row (drive voltages) is not used.

    A frame is marked when its timestamp falls in an interval ``(on, off]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

EventFrames = Tuple[Optional[np.ndarray], List[str]]


def led_csv_frames(movie_path: str | Path, num_frames: int) -> EventFrames:
    """Mark the frames acquired while the LED was on (``led_csv`` format).

    Returns ``(flags, notes)``. ``flags`` is None when the movie has no LED
    files (no perturbation recorded) or they can't be used; ``notes`` explains
    the latter, and any mismatch between the timings and the movie.
    """
    path = Path(movie_path)
    stem = path.with_suffix("")
    frame_file = stem.parent / f"{stem.name}.csv"
    led_file = stem.parent / f"{stem.name} led.csv"
    has_frames, has_led = frame_file.is_file(), led_file.is_file()
    if not has_frames and not has_led:
        return None, []
    if not (has_frames and has_led):
        missing = frame_file if not has_frames else led_file
        return None, [f"blue LED: {missing.name!r} missing; blue plane left empty"]

    try:
        times = np.loadtxt(frame_file, delimiter=",", ndmin=1).ravel()
        switches = np.loadtxt(led_file, delimiter=",", ndmin=2)[0]
    except ValueError as exc:
        return None, [f"blue LED: could not read timing files ({exc}); blue plane left empty"]

    notes: List[str] = []
    if len(switches) % 2:
        notes.append(f"blue LED: odd number of switch times ({len(switches)}); "
                     f"the last one is ignored")
    flags = np.zeros(len(times), dtype=bool)
    for on, off in zip(switches[0::2], switches[1::2]):
        flags |= (times > on) & (times <= off)

    if len(times) != num_frames:
        extra = len(times) > num_frames
        notes.append(f"blue LED: {len(times)} frame timestamps for {num_frames} frames; "
                     f"{'extra timestamps ignored' if extra else 'frames without one left unmarked'}")
        padded = np.zeros(num_frames, dtype=bool)
        padded[:min(len(flags), num_frames)] = flags[:num_frames]
        flags = padded
    return flags, notes


_SOURCES: Dict[str, Callable[[str | Path, int], EventFrames]] = {
    "led_csv": led_csv_frames,
}

#: Valid values of ``[events] source``.
EVENT_SOURCES = ("none",) + tuple(_SOURCES)


def load_event_frames(source: str, movie_path: str | Path, num_frames: int) -> EventFrames:
    """Per-frame event flags for ``movie_path`` from the named ``source``.

    ``"none"`` returns ``(None, [])``, so callers need no special case.
    """
    if source == "none":
        return None, []
    if source not in _SOURCES:
        raise ValueError(f"Unknown event source {source!r}; expected one of {EVENT_SOURCES}.")
    return _SOURCES[source](movie_path, num_frames)
