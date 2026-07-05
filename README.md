# optomerge

Excise and register multichannel sub-images from raw fluorescence movie stacks.

Two-colour (and single-colour) TIRF/fluorescence movies often pack several
imaging channels into different regions of the same camera frame. `optomerge`
finds those channels automatically, registers them onto a common reference
(sub-pixel FFT phase correlation with a scale/rotation search), background-
subtracts and normalises them, and writes an aligned, false-coloured RGB movie.
It is the algorithmic equivalent of the MATLAB `align` toolbox (Ruijgrok /
Nakamura), reimplemented in Python with an object-oriented API.

## Quick start

```bash
git clone git@github.com:paulruijgrok/optomerge.git
cd optomerge
python3 -m venv .optomerge && source .optomerge/bin/activate
pip install -e optomerge/
python run_optomerge.py                      # process everything under test_data/
```

Expected output: an aligned RGB TIFF for each input movie, written under
`output_temp/` with the input folder structure mirrored and `_aligned` appended
to each filename, plus a `run_log.txt` summarising the batch. Use
`python run_optomerge.py --dry-run` first to see what would be processed.

## Installation

### Requirements

- Python 3.9+ (developed and tested on 3.11)
- `numpy`, `scipy`, `scikit-image`, `tifffile` (installed automatically)
- `tomli` — installed automatically only on Python 3.9/3.10, to read TOML config
  files (Python 3.11+ uses the standard-library `tomllib` instead)
- Optional: `matplotlib` for the diagnostic script, `pytest` for the test suite
  (`pip install -e "optomerge/[dev]"`), `opencv-python` for a faster warp/blur
  backend (`pip install -e "optomerge/[cv2]"`)

### Setup

The installable package lives in the nested `optomerge/` directory (which holds
`pyproject.toml`), so install it by pointing pip at that folder:

```bash
python3 -m venv .optomerge
source .optomerge/bin/activate
pip install -e "optomerge/[dev]"
pytest optomerge/tests/          # optional: verify the install
```

### Known gotchas

- **Install target is `optomerge/`, not `.`** — the package is at
  `optomerge/optomerge/`; run `pip install -e optomerge/`.
- **Virtualenvs are not relocatable.** A venv hard-codes absolute paths, so
  renaming or moving the repo folder breaks the venv's `pip`/`activate` (they
  keep pointing at the old path and silently fall back to base Python). If you
  move the repo, delete and recreate the venv rather than copying it.

## Usage

### Library API

The public API is object-oriented; `MergePipeline` wires a `RawMovie` through a
`ChannelLayout`, an `Aligner`, and out to an `RGBMovie`:

```python
from optomerge import MergePipeline, ChannelLayout, PhaseCorrelationAligner

pipe = MergePipeline(
    "movie.tif",
    layout=ChannelLayout.auto(),               # or .top_green_bottom_red(), etc.
    aligner=PhaseCorrelationAligner(),          # use_scrub=True for sub-pixel
    projection_frames=200,                       # frames used for detection
    bg_radius=10,
)
rgb = pipe.run(output="movie_aligned.tif")       # RGBMovie; also saved to disk
```

See `optomerge/DESIGN.md` for the full object model (Movie/Frame/Channel/
Layout/Transform/Aligner and the pluggable reader/writer seams). The validated
numerical routines live in the private `optomerge/optomerge/_kernels/`
subpackage and are reached only through the public classes.

### Command-line scripts

- **`run_optomerge.py`** — batch processor (see below).
- **`test_pipeline.py`** — step-by-step diagnostic: runs one movie through each
  stage and saves a PNG after every step (projection, channel segmentation,
  cropped channels, before/after alignment, first RGB frame). Useful for
  eyeballing why a particular movie aligned the way it did.
- **`profile_pipeline.py`** — times each pipeline stage on one movie and prints
  a ranked breakdown of where the time goes.

Every script takes `--help`.

## Batch processing

`run_optomerge.py` is the unattended batch runner. It recursively finds every
`.tif`/`.tiff` under `--input` (default `test_data/`), processes each into
`--output` (default `output_temp/`) with the folder structure mirrored, and is
built to survive an overnight run:

- **Fail-isolated** — one movie's failure is logged with a traceback and the
  batch continues.
- **Resumable** — movies whose output already exists are skipped unless
  `--overwrite` is given.
- **Logged** — a per-batch `run_log.txt` records every file, its fitted
  transform, and a success/failure summary.

```bash
python run_optomerge.py --channel-order auto --frames 200
python run_optomerge.py --dry-run                 # list files, do nothing
```

### Robust alignment, acceptance & conserving across a set

Aligning two structurally-different channels is finicky — some movies align
cleanly and others don't. Three features address this:

- **Acceptance criteria** reject nonsensical fits (a channel far too small, or a
  transform whose translation / rotation / scale is implausibly large). A
  rejected movie is logged and skipped, not saved. Thresholds live in the
  `[acceptance]` config section.
- **Robust calibrator** (`--robust`) projects the movie in chunks, fits + scores
  each, keeps only candidates that pass acceptance, and picks the best of N —
  robust to a bad stretch of frames.
- **Constrained peak search** (`--max-shift PX`) restricts the registration to a
  small translation window, so a spurious far-off correlation peak can't win.
  Use when the true inter-channel shift is small (two halves of one camera frame).

When one movie in a session aligns well and others don't, **conserve** that
alignment across a set: compute it once and apply it to the rest.

```bash
# Each _ch\d\d_ group's first movie defines the alignment; the rest reuse it
python run_optomerge.py --reuse-alignment first --group-by token --robust --max-shift 30

# Use one known-good movie as a predetermined reference for the whole batch
python run_optomerge.py --reuse-alignment "good_movie.tif" --max-shift 30
```

Sets are partitioned by `--group-by run|token|folder`; intensity normalisation
is always recomputed per movie, so only the geometric alignment is shared.

### Configuration & run provenance

Run parameters resolve in three layers, each overriding the previous: built-in
defaults → `--config` TOML file(s) → explicit CLI flags. See
`optomerge.example.toml` for the full annotated schema (`optomerge.config.Settings`).

```bash
python run_optomerge.py --config my_run.toml --channel-order auto   # file + CLI override
```

Every run writes its fully-resolved configuration to `<output>/run_config.toml`,
so any result is reproducible by feeding that file straight back in:
`python run_optomerge.py --config output_temp/run_config.toml`. TOML config works
on Python 3.9+ — 3.11+ uses the standard-library `tomllib`, and on 3.9/3.10 the
`tomli` backport is installed automatically.

## Repo map

- `optomerge/` — the installable package
  - `optomerge/optomerge/` — object-oriented public API
  - `optomerge/optomerge/_kernels/` — private numerical core (I/O, segmentation,
    registration, transform, processing)
  - `optomerge/tests/` — test suite (`pytest optomerge/tests/`)
  - `optomerge/DESIGN.md` — architecture / object model
  - `optomerge/pyproject.toml` — package metadata and dependencies
- `run_optomerge.py`, `test_pipeline.py`, `profile_pipeline.py` — entry-point scripts
- `test_data/` — sample movies (git-ignored)
- `output_temp/` — script output (git-ignored)

## Status

The object-oriented `optomerge` package is the single, current implementation;
the earlier procedural package and its OOP proposal have been consolidated into
it. The full test suite passes, including an end-to-end pipeline test on the
sample movies.

Registration currently supports one reference plus one moving channel via phase
correlation, with acceptance criteria, a robust best-of-N calibrator, a
constrained peak search, and conserve-alignment across movie sets (see above).
Channels are segmented by default with a robust row-intensity-profile method
that finds tight axis-aligned rectangles (`--segmentation row_profile`; the
legacy k-means/slanted-line search is `line_search`). The layout / aligner /
reader / writer / calibrator / segmentation seams are designed to make
additional channel arrangements, methods, and file formats drop-in extensions.

Known limitations / future work:

- **`--channel-order auto` is unreliable** — it often fails to split the two
  stacked channels; forcing an explicit order is recommended for now.
- **Output normalisation** uses each channel's projection min/max, which can
  make bright frames look over-saturated; robust (percentile) normalisation is
  the next planned improvement.
- Not yet ported from the procedural pipeline: GPU acceleration and file-level
  parallelism. Predetermined alignment can be taken from a reference movie
  (`--reuse-alignment FILE`); saving/loading a fitted alignment to a sidecar
  file is not yet implemented.
