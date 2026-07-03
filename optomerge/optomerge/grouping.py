"""
optomerge.grouping
==================
Partition a list of movies into *sets* that should share one alignment.

When alignment is conserved across a set, one calibration (channel bounds +
transform) is computed on the set's first movie and applied to the rest. How
movies are partitioned into sets is a pluggable choice:

* ``"run"``    — every movie is one set (the whole batch shares an alignment).
* ``"token"``  — group by a regex token in the filename stem (e.g. ``_ch\\d\\d_``,
  the channel/date marker from ``alignRGB.m``'s ``keepAlignmentPattern``): movies
  whose stem yields the same token value form a set.
* ``"folder"`` — group by containing directory.

New strategies are added by registering another key function in ``_GROUPERS``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

# Default channel/date marker, as in alignRGB.m's keepAlignmentPattern.
DEFAULT_TOKEN_PATTERN = r"_ch\d\d_"


def _key_run(path: Path, token_pattern: str) -> str:
    return "all"


def _key_folder(path: Path, token_pattern: str) -> str:
    return str(path.parent)


def _key_token(path: Path, token_pattern: str) -> str:
    match = re.search(token_pattern, path.stem)
    # No match -> the movie is its own singleton set (keyed by its stem).
    return match.group(0) if match else f"__notoken__:{path.stem}"


_GROUPERS: dict[str, Callable[[Path, str], str]] = {
    "run": _key_run,
    "folder": _key_folder,
    "token": _key_token,
}


def available_modes() -> List[str]:
    """Return the registered grouping mode names."""
    return list(_GROUPERS)


def group_movies(
    paths: Sequence,
    by: str = "run",
    token_pattern: str = DEFAULT_TOKEN_PATTERN,
) -> List[Tuple[str, List[Path]]]:
    """Partition ``paths`` into ordered sets.

    Parameters
    ----------
    paths : sequence of path-like
        Movie paths, already in processing order.
    by : str
        Grouping strategy: one of :func:`available_modes` (default ``"run"``).
    token_pattern : str
        Regex used by the ``"token"`` strategy.

    Returns
    -------
    list of (group_key, [Path, ...])
        Groups in first-seen key order; paths within a group keep input order.
    """
    try:
        keyfn = _GROUPERS[by]
    except KeyError:
        raise ValueError(
            f"Unknown grouping mode {by!r}; choose from {available_modes()}"
        )

    groups: dict[str, List[Path]] = {}
    order: List[str] = []
    for p in paths:
        p = Path(p)
        key = keyfn(p, token_pattern)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(p)
    return [(key, groups[key]) for key in order]
