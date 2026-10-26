# 2026-10-26 — v3d sign-off, merge to main; blue-LED perturbation marking

Notion mirror: *2026/10/26 v3d sign-off, merge to main* under the Optomerge log
hub. Session on Sherlock (`sh03-ln08`, tests in a `possu` job), branch
`feature/sherlock`.

## TL;DR

- **P. signed off `aligned_feature_v3d`:** registration, intensity balance and
  crop size all look good on the pilot. These are the current defaults (all
  2026-10-24 fixes plus `norm_mode = "objects"`).
- **`norm_target = 0.5` is kept** as the brightness default.
- **Docs:** `docs/general_registration.md` now marks the July ECC/MI sweep as
  superseded and adds the 2026-10-24 pilot accuracy comparison (`36310d5`).
- **Merged `feature/sherlock` → `main`** (fast-forward, 17 commits) and pushed
  both branches. Tests: 169 passed, 1 skipped.
- `aligned_feature_v3d` is **not** copied off `$SCRATCH`. P. will regenerate:
  more of the dataset is being uploaded, and everything will be reprocessed in
  one go once the algorithm is final.

## What changed and why

### Docs (`36310d5`)
- The July comparison in `docs/general_registration.md` predates two fixes:
  - ECC/MI returned identity on unequal crops (now `fit_in_canvas`).
  - MI's coarse search had its sign reversed (`bd722ff`).

  Those rows are now flagged as historical.
- A new section has the pilot accuracy table (feature 0.24 px median, 0/27 > 1 px;
  ECC 1.10 px; MI 1.59 px). It comes from the 2026-10-24 journal.
- The convention note said "mismatched shapes return identity". It now explains
  how unequal crops are handled.

### Merge
`main` had no commits of its own, so `git merge --ff-only feature/sherlock`.

## Commands

```bash
# tests (in a job)
srun -p possu -c 4 --mem=16G -t 00:20:00 bash -c 'source hpc/env.sh; python -m pytest -q'
# -> 169 passed, 1 skipped (job 49664370)

git checkout main && git pull --ff-only origin main
git merge --ff-only feature/sherlock
git push origin main feature/sherlock
```

## Files / commits

| Commit | What |
|---|---|
| `36310d5` | `docs/general_registration.md`: July ECC/MI sweep superseded; pilot accuracy comparison |
| `92b4ffe..36310d5` | `main` fast-forwarded to `feature/sherlock` |
| `eb6f2ae` | `journals/2026-10-26_v3d-signoff-merge-main.md` (part 1) |
| `5fc5442` | `src/optomerge/events.py` (new), `config.py` (`[events]`), `movie.py`, `pipeline.py`, `__init__.py`, `run_optomerge.py` (`--events`), README, `optomerge.example.toml`, `tests/test_events.py` |
| `f8b408f` | `.gitignore`: `old_code/` |
| (this commit) | this journal (part 2) |

## Part 2: blue-LED perturbation marking (`5fc5442`, `f8b408f`)

P. asked for the output's blue plane to show when the external perturbation
(blue LED) was on, as the old MATLAB pipeline did (`getBlueLEDtimings` in
`alignRGB.m`, placed locally in `old_code/`). It should be optional for general
users but work for this data.

### Format (checked on all 55 timing CSVs in the pilot data)
- `<stem>.csv`: one acquisition timestamp (ms) per frame, 800 lines.
- `<stem> led.csv`: row 1 has the switch times; row 2 has the drive voltages
  (`0.000,1.100`, as P. confirmed).
  - MATLAB marks frames in `(on, off]` pairs from row 1 and ignores row 2.
  - P. confirmed that reading is correct.
- Every pilot file has the same protocol: one pulse, frames 100–299 (0-based)
  of 800.
- `20180620/ch08 … 05 led.csv` has no frame-timing CSV.
- The old composites have blue = 128 on exactly those frames and 0 elsewhere.

### Implementation
- **`src/optomerge/events.py`:**
  - Each setup-specific format is a named source behind `load_event_frames()`.
    The first one is `led_csv`.
  - No files → unmarked, with no warning.
  - Anything else goes to a `[QC]` warning and never fails the movie: one file
    missing, unreadable, an odd number of switches, or a timestamp/frame count
    mismatch.
  - MATLAB's "copy the previous movie's pattern" fallback is not ported. It
    depends on file order, and a misspelt variable meant it never took effect.
- **Config:** new `[events]` section with `source` (default `"none"`) and
  `blue_level` (default 0.5). The CLI flag is `--events`.
  - `RGBMovie.from_channels` and `MergePipeline.merge` take `blue_level`.
  - The blue fill is vectorised.
- README section and `optomerge.example.toml` updated.
- `old_code/` is gitignored at P.'s request (not part of the repo).
- `hpc/runs/MyLOV1Char/base.toml` now sets `[events] source = "led_csv"`, so the
  full reprocessing run marks the perturbation.

### Verification
- Full suite (job 49667441): 177 passed, 1 skipped (8 new tests in
  `tests/test_events.py`).
- Real data (job 49667857): the 4 movies of 20181002/ch11 with
  `--events led_csv`, output in
  `$SCRATCH/optomerge_runs/MyLOV1Char/aligned_blue_test/`.
  - Blue is on for frames 100–299, uniform, and matches the old composites frame
    for frame.
- **Level:** blue is 127, not 128, because the 8-bit writer floors
  (`(x*255).astype(uint8)`). P. accepted 127, so the writer stays as it is.

```bash
srun -p possu -c 8 --mem=32G -t 00:30:00 bash -c "source hpc/env.sh; \
  python run_optomerge.py --config hpc/runs/MyLOV1Char/base.toml hpc/runs/MyLOV1Char/20181002.toml \
  --input $IN/20181002/ch11 --output $SCRATCH/optomerge_runs/MyLOV1Char/aligned_blue_test \
  --events led_csv --workers 3 --overwrite"
```

Merged `feature/blue-led` → `main` (fast-forward) and pushed both branches.

## Open items / next steps

- **P. is uploading more of the dataset.** Once the algorithm is final, reprocess
  everything in one batch run (SLURM, `--num-shards`/`--workers`). Size the job
  with a small representative test first.
- ~~Delete the clone `~/code/optomerge-objnorm`~~ (done: P. removed it).
- **`reuse_alignment = "first"`:** store the transform in a crop-independent
  (raw-frame) form.
- Range mode only: the `vmax` cap misses outliers larger than 0.1% of the channel.
- Possibly fold the diagnostic into a general `diagnose_run.py`.
- Carried over: `--channel-order auto`, sidecar alignment save/load, GPU,
  Deep-LASI validation, stale `SPEEDUP_PROPOSAL.md`.
