# 2026-10-10/11 — Sherlock setup, input exclusion, MyLOV1Char pilot

Notion mirror: *2026/10/10 Sherlock setup and MyLOV1Char pilot* under the
Optomerge log hub. Written up retrospectively on 2026-10-24 from the commits, the
job records and the pilot logs.

## TL;DR

- **Sherlock environment** is built and working. It lives in
  `$GROUP_HOME/envs/optomerge`, with compiled dependencies from conda-forge,
  installed through the group miniconda. `hpc/env.sh` finds it automatically.
- **The skill is tracked in the repo** (`.claude/skills/pipeline-engineering-standards/`),
  and `CLAUDE.md` now describes the Mac and Sherlock as one project synced through
  GitHub.
- **New opt-in input exclusion**: `[io] exclude = [globs]` / `--exclude GLOB`. The
  MyLOV1Char tree holds older derived files (`* RGB.tif`, fils/ridge/heads) next to
  the raw movies.
- **The channel order differs by recording date** in MyLOV1Char, so there is one
  config per date layered on a shared `base.toml`.
- **The pilot ran on 2026-10-11** (job `47397242`, `possu`, 32 CPU / 128 GB):
  - all **27/27 movies succeeded**, with 0 rejected and 0 failed
  - **2:39 wall time**, about 30 s per movie
  - MaxRSS **34.6 GB** with 8 workers, so about 4.3 GB per movie
- Tests: 154 passed, 1 skipped. Everything is pushed on `feature/sherlock`, which
  is not yet merged to `main`.

## What changed and why

### Environment (`a2e528a`)
- **Wheels don't work.** CentOS 7's glibc is too old for current scipy/OpenCV
  wheels. `pip install -e .` falls back to building scipy from source, which fails
  without OpenBLAS.
- **The classic conda solver doesn't either.** The classic solver in the possu base
  install runs out of memory at 16 GB on this solve.
- **What works:** the group miniconda (libmamba) builds the env from conda-forge as
  `$GROUP_HOME/envs/optomerge`, then optomerge is installed with `pip install
  --no-deps -e .`.
  - The build ran inside dev job `47199732`, which took 11 minutes; pytest passed
    there (153 passed, 1 skipped).
- **`hpc/env.sh`:** `CONDA_ENV` may now be a prefix path and `CONDA_BASE` can be
  overridden. On Sherlock it auto-detects `$GROUP_HOME/envs/optomerge`.
- **`hpc/README.md`:** has a Sherlock section with the recipe above.
  - TOML files can't expand `$SCRATCH`, so they need written-out paths.
  - `pip check` complains that `opencv-python-headless` is missing. This is
    harmless: conda's `opencv` provides `cv2` under a different package name.

### One project, two machines (`44827ef`)
- The pipeline-engineering-standards skill, uploaded from the Mac, is now tracked
  in `.claude/skills/`. Both machines load the same version after a pull.
- New `CLAUDE.md` section: the Mac and Sherlock are one project synced through
  GitHub, and nothing that belongs to the project should be kept only on one
  machine.
- `.gitignore` now ignores macOS AppleDouble files (`._*`).
- Tests passed in dev job `47242582`.

### Input exclusion (`5bfe199`)
- **Why:** discovery picked up every `.tif`, so the derived files in MyLOV1Char
  would have been treated as input movies.
- **What:** `IOSettings.exclude` is a list of filename globs, matched with
  `fnmatchcase` against the file name only.
  - The default is `[]`, so it's opt-in and nothing changes for existing configs.
  - The CLI `--exclude` replaces the config list rather than adding to it, like the
    other CLI overrides.
  - The active patterns are written to the run log header.
- **Test:** `test_exclude_globs_skip_derived_files`.
- **Dry run on MyLOV1Char with `["*RGB*"]`:** 27 raw movies kept, 60 derived files
  excluded.

### MyLOV1Char pilot configs (`96173fe`)
- **How the channel order was found:** each raw movie's two halves were
  template-matched against the movie's earlier `* RGB.tif` composite, using its
  green channel. The matching half scored 0.80–0.99; the other half scored
  0.14–0.42.

  | Date | Movies | Channel order |
  |---|---|---|
  | 20170803 | 6 | `top_red_heads_bottom_green_fils` |
  | 20180620 | 14 | `top_green_fils_bottom_red_heads` |
  | 20181002 | 7 | `top_red_heads_bottom_green_fils` |

- **Config layout:** `base.toml` sets the feature aligner with `fit_rotation`,
  `exclude ["*RGB*"]` and `reuse_alignment "none"`. Each per-date TOML adds the
  input path, output path and channel order.
  - Outputs go to `$SCRATCH/optomerge_runs/MyLOV1Char/aligned_feature/<date>`.
- **`run.sbatch`:** runs the three dates in sequence on one node, 8 movies × 4
  threads at a time.
  - A failed date doesn't stop the others.
  - It logs the git hash and library versions.
- The old `hpc/sherlock_test.toml` was dropped, because it assumed one channel
  order for the whole tree.

### Pilot results (job `47397242`, 2026-10-11 10:58–11:00)
- **Node and stack:** `sh03-01n65`, repo @ `96173fe`. numpy 2.4.6, scipy 1.17.1,
  cv2 5.0.0, skimage 0.26.0.
- **Results by date:**

  | Date | Movies | Succeeded | Time |
  |---|---|---|---|
  | 20170803 | 6 | 6 | 32 s |
  | 20180620 | 14 | 14 | 60 s |
  | 20181002 | 7 | 7 | 30 s |

- **Resources:**
  - Elapsed 2:39 and TotalCPU 20:01, about 24% of the 32 cores.
  - Cores sat idle because the dates run one after another and each date has only
    6–14 movies.
  - MaxRSS 34.6 GB.
- **Fitted transforms:** the red channel is shifted by about (−0.5…0.7,
  −9…−12) px with rotation ≤ 0.16°. The feature score is 0.27–0.43 for 20170803,
  lower for ch09 than for ch08.

## Commands

```bash
# env build (dev job 47199732)
srun -p dev -c 4 --mem=16G -t 00:30:00 bash -c '
  export CONDA_PKGS_DIRS=$SCRATCH/cache/conda_pkgs
  $GROUP_HOME/miniconda/bin/conda create -y -p $GROUP_HOME/envs/optomerge \
      -c conda-forge --override-channels python=3.11 numpy scipy scikit-image \
      tifffile opencv pytest matplotlib
  source $GROUP_HOME/miniconda/etc/profile.d/conda.sh
  conda activate $GROUP_HOME/envs/optomerge
  pip install --no-deps -e . && pytest -q'

# dry runs, one per date
python run_optomerge.py --config hpc/runs/MyLOV1Char/base.toml hpc/runs/MyLOV1Char/20180620.toml --dry-run

# pilot
sbatch hpc/runs/MyLOV1Char/run.sbatch
sacct -j 47397242 -o JobID,State,Elapsed,MaxRSS,TotalCPU
```

## Files / commits

| Commit | What |
|---|---|
| `a2e528a` | `hpc/env.sh`, `hpc/README.md` (Sherlock section), `hpc/sherlock_test.toml` |
| `44827ef` | `.claude/skills/pipeline-engineering-standards/`, `CLAUDE.md`, `.gitignore` |
| `5bfe199` | `IOSettings.exclude` (`config.py`), `--exclude` (`run_optomerge.py`), `optomerge.example.toml`, `tests/test_sharding.py`; drops `sherlock_test.toml` |
| `96173fe` | `hpc/runs/MyLOV1Char/{base,20170803,20180620,20181002}.toml`, `run.sbatch`, `hpc/README.md` |

Pilot outputs and logs are in `$SCRATCH/optomerge_runs/MyLOV1Char/`
(`slurm_47397242.out`, `aligned_feature/<date>/run_log.txt`).

## Open items / next steps

- **Visual review (done 2026-10-24 by P.):** no alignment is badly wrong and many
  are very good. Some are visibly off, possibly more so in movies with sparse
  filaments. For example, in the first 20170803 movie the red channel is offset
  towards the lower right. The green/red intensity balance is good in most movies
  but off in a few. Analysis follows in the 2026-10-24 journal.
- **Merge `feature/sherlock` → `main`** after the write-up (full pytest, PR).
- **Sizing:**
  - Memory is about 4.3 GB per movie. `_common.sh` assumes 5 GB, which is safe.
  - For larger trees, run the dates in parallel (a job array, or shards) rather
    than in sequence to use the idle cores.
  - A worker-count sweep is still to do.
- **Redo with other aligners** (e.g. `mi`) into `aligned_<method>/` to compare.
- `._*` / `.DS_Store` files have appeared in the `$SCRATCH` output tree, from
  browsing it on the Mac. `._*.tif` inputs would be picked up by discovery; consider
  excluding them by default.
- Carried over from 2026-10-09:
  - `--channel-order auto` is unreliable
  - sidecar save/load of a fitted alignment
  - tighter transform padding
  - GPU support
  - Deep-LASI validation
  - `SPEEDUP_PROPOSAL.md` is stale
