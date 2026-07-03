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
python run_optomerge.py --reuse-alignment first   # reuse one movie's alignment across the batch
python run_optomerge.py --dry-run                 # list files, do nothing
```

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

Not yet ported from the older procedural pipeline: TOML config files, GPU
acceleration, file-level parallelism, and the multi-chunk "robust" alignment
mode. Registration currently supports one reference plus one moving channel via
phase correlation; the layout/aligner/reader/writer seams are designed to make
additional channel arrangements, alignment strategies, and file formats
drop-in extensions.
```
