"""
optomerge._progress
===================
Lightweight, dependency-free progress display for the command line.

Two display modes are provided:

``ProgressBar``
    In-place updating bar for slow, frame-counted loops (background
    subtraction, affine transform).  Uses ``\\r`` to overwrite the current
    line when stdout is a TTY; falls back to milestone lines (0 / 25 / 50 /
    75 / 100 %) when output is redirected.

    Example output (TTY)::

        Bg subtraction (green)    [##########--------------------------]  28%   (224/800 frames)

``step_line``
    Single printed line for fast steps that do not have per-item progress
    (loading, projection, channel finding, alignment).

    Example output::

        Loading frames                                              0.31 s
        Finding channel bounds                                      0.04 s

Usage
-----
.. code-block:: python

    from optomerge._progress import ProgressBar, step_line

    bar = ProgressBar(800, label="Bg subtraction (green)")
    for i in range(800):
        process(i)
        bar.advance()
    bar.done(elapsed=12.4)

    step_line("Loading frames", elapsed=0.31)
"""

from __future__ import annotations

import sys
import threading
import time


# ---------------------------------------------------------------------------
# TTY detection
# ---------------------------------------------------------------------------

def _is_tty() -> bool:
    """Return True when stderr is an interactive terminal."""
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


# ---------------------------------------------------------------------------
# ProgressBar
# ---------------------------------------------------------------------------

class ProgressBar:
    """Thread-safe, in-place ASCII progress bar written to stderr.

    Parameters
    ----------
    total : int
        Total number of items (frames).
    label : str
        Short description displayed to the left of the bar (max ~32 chars).
    width : int
        Width of the ``[####----]`` bar section in characters.  Default 36.
    show_count : bool
        Show ``(N / M frames)`` suffix after the percentage.  Default True.
    unit : str
        Noun used in the count suffix.  Default ``'frames'``.
    """

    BAR_FILL  = "#"
    BAR_EMPTY = "-"

    def __init__(
        self,
        total: int,
        label: str = "",
        width: int = 28,
        show_count: bool = True,
        unit: str = "frames",
    ) -> None:
        self.total      = max(1, total)
        self.label      = label
        self.width      = width
        self.show_count = show_count
        self.unit       = unit

        self._lock       = threading.Lock()
        self._completed  = 0
        self._last_pct   = -1          # last percentage rendered
        self._tty        = _is_tty()

        # Print the empty bar immediately so the user sees the label
        self._render(0)

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def advance(self) -> None:
        """Increment the completed count by one and refresh the display."""
        with self._lock:
            self._completed = min(self._completed + 1, self.total)
            self._render(self._completed)

    def done(self, elapsed: float | None = None) -> None:
        """Force the bar to 100 % and print a trailing newline.

        Parameters
        ----------
        elapsed : float or None
            Wall-clock seconds for the step; printed after the bar if given.
        """
        elapsed_str = f"  {elapsed:6.1f} s" if elapsed is not None else ""
        with self._lock:
            if self._tty:
                # Overwrite the current line with the final 100 % bar,
                # then append the elapsed time on the same line.
                self._render(self.total, force=True)
                print(elapsed_str, file=sys.stderr, flush=True)
            else:
                # Non-TTY: only print the final line if we haven't already
                # rendered 100 % as a milestone (avoids a duplicate line).
                if self._last_pct < 100:
                    self._render(self.total, force=True)
                print(f"  done{elapsed_str}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ #
    #  Internal rendering                                                  #
    # ------------------------------------------------------------------ #

    def _render(self, n: int, force: bool = False) -> None:
        """Render the bar for *n* completed items.

        Called with the lock already held (or during __init__).
        """
        pct = int(100 * n / self.total)

        # In TTY mode re-render on every percent change; outside TTY only
        # at 0 / 25 / 50 / 75 / 100 to avoid flooding log files.
        if not force:
            if pct == self._last_pct:
                return
            if not self._tty and pct % 25 != 0:
                return

        self._last_pct = pct

        filled    = int(self.width * n / self.total)
        bar_str   = self.BAR_FILL * filled + self.BAR_EMPTY * (self.width - filled)
        count_str = f"  ({n}/{self.total} {self.unit})" if self.show_count else ""
        line      = f"  {self.label:<36s}  [{bar_str}] {pct:3d}%{count_str}"

        if self._tty:
            # Overwrite current line
            print(f"\r{line}", end="", flush=True, file=sys.stderr)
        else:
            # Append a new line for log files
            print(line, flush=True, file=sys.stderr)


# ---------------------------------------------------------------------------
# step_line — fast-step display
# ---------------------------------------------------------------------------

def step_line(label: str, elapsed: float | None = None) -> None:
    """Print a single status line for a fast step (no per-item progress).

    Parameters
    ----------
    label : str
        Short description of the step.
    elapsed : float or None
        Wall-clock seconds; printed right-aligned if given.

    Example output::

        Loading frames                                              0.31 s
    """
    elapsed_str = f"{elapsed:6.1f} s" if elapsed is not None else ""
    # Use a fixed-width layout so columns align across steps
    print(f"  {label:<36s}  {elapsed_str}", flush=True, file=sys.stderr)
