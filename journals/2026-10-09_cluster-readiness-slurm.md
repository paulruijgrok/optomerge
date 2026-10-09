# 2026-10-09 — Cluster readiness: merges, SLURM job arrays, journal backfill

Notion mirror: *2026/10/09 Cluster readiness: merges, SLURM job arrays, journal
backfill* under the Optomerge log hub.

## TL;DR

- Reviewed the project state after three months idle; the last activity was
  2026-07-11. Decision: the pipeline is ready for a first cluster test with the
  **feature aligner**, and runs must be cheap to redo with other settings.
- **Repo tidy-up and merges into local `main`:**
  - Local `main` was stale. The perf branch had already been merged on GitHub as
    PR #5 (07-10), so local `main` was fast-forwarded to it.
  - Untracked duplicates of perf files and three shelved tests were removed. They
    were byte-identical to the committed versions.
  - Two stale `.git/*.lock` files from July were removed.
  - `feature/general-aligners` was rebased onto `main`, its docs committed, and the
    branch merged.
- **SLURM support** (`feature/slurm-hpc`, merged), adapted from FASTrack's `hpc/`:
  - `--num-shards/--shard-index`
  - atomic outputs
  - CPU-allocation-aware threading
  - `hpc/` sbatch scripts and a guide
- **Tests:** 154 pass (13 new). A simulated two-task array run worked end to end.
- **Journaling:** backfilled 2026-07-10/11 and started a `journals/` folder. Added
  a `CLAUDE.md` with the journaling SOP.
- **Not pushed.** The session's shell cannot authenticate to GitHub. Push from the
  Mac (see below).

## What changed and why

### Branch / repo state
- Found `main` at `495b164` locally while `origin/main` was at `406d3c2` (PR #5,
  the perf work, merged 07-10). Fast-forwarded.
- `feature/general-aligners` (`bca13f4`, never pushed) rebased onto `main` with no
  conflicts. Commit `70d4f83` adds `docs/candidate_datasets.md` and a 31-movie
  comparison section to `docs/general_registration.md`. It was merged as `b1943b6`.
  Worker processes rebuild the calibrator from `Settings`, so ecc/mi/keypoint also
  work with `--workers`.
- `docs/skills/` (a copy of the pipeline-engineering-standards skill) is still
  untracked. It's for P. to decide whether to keep it.

### SLURM support (`3f5c2bc`, merged as `b651766`)
P. asked to reuse FASTrack's SLURM scripts. FASTrack shards a *manifest of
datasets*. optomerge's unit of work is a *movie inside one input tree*, so
sharding went into the runner itself:

- **`--num-shards N --shard-index I`**: `grouping.select_shard` deals the sorted
  movies out round-robin.
  - With `--reuse-alignment first` it deals out whole sets instead, because the
    first movie seeds the rest.
  - All shards write into one output tree, with per-shard logs
    `run_log_shardNNNofMMM.txt`.
  - `run_config.toml` is written atomically, and the shard fields are reset so it
    is identical across shards.
  - Empty shards exit 0.
- **Atomic outputs**: `save_rgb_tiff` now writes `.<name>.partial` and renames it
  when complete. Without this, a preempted or timed-out job could leave a
  truncated `.tif` that resume skips as finished.
- **`available_cpus()`** (affinity mask, capped by `SLURM_CPUS_PER_TASK`) replaces
  `os.cpu_count()` in three places: the per-frame pools, the worker thread cap, and
  cv2's pool. On a shared node `os.cpu_count()` sees every core, so jobs would
  oversubscribe their allocation.
- **`hpc/`** contains:
  - `env.sh` (conda env `optomerge`)
  - `_common.sh`: finds the repo via `$SLURM_SUBMIT_DIR`, because sbatch runs a
    spool copy and `$0` is useless. It sets `WORKERS = min(cpus, mem_GB / 5)`.
  - `optomerge_single.sbatch` (pilot)
  - `optomerge_array.sbatch` (job array)
  - `example_cluster.toml` (feature + `fit_rotation`, explicit channel order,
    `reuse_alignment = "none"`)
  - `README.md`
- Defaults: 16 CPUs, 64 GB, 4 h, partition `possu`, `--requeue`.

## Commands

```bash
# repo hygiene (Linux VM shell on the Mac's repo folder)
rm .git/HEAD.lock .git/index.lock     # stale, from 2026-07-11
git checkout main && git merge --ff-only origin/main
git checkout feature/general-aligners && git rebase main
git checkout main && git merge --no-ff feature/general-aligners
git checkout -b feature/slurm-hpc ... && git checkout main && git merge --no-ff feature/slurm-hpc

# tests (fresh venv, Python 3.10, opencv-python-headless 5.0)
python -m pytest -q                   # 154 passed

# simulated 2-task array (fake SLURM env + stub conda), synthetic movies
SLURM_SUBMIT_DIR=<repo> SLURM_ARRAY_TASK_MIN=0 SLURM_ARRAY_TASK_MAX=1 \
SLURM_ARRAY_TASK_ID=0 SLURM_CPUS_PER_TASK=2 SLURM_MEM_PER_NODE=12288 \
  bash hpc/optomerge_array.sbatch run.toml
```

**To publish (on the Mac):**

```bash
cd ~/Documents/Claude/Projects/optomerge
git push origin main feature/general-aligners feature/slurm-hpc
```

**First cluster test:**

```bash
git clone https://github.com/paulruijgrok/optomerge.git && cd optomerge
conda create -n optomerge python=3.11 -y && conda activate optomerge && pip install -e .
cp hpc/example_cluster.toml my_run.toml       # set absolute input/output paths
sbatch hpc/optomerge_single.sbatch my_run.toml   # pilot on the 31 test movies
sacct -j <jobid> -o JobID,Elapsed,MaxRSS,AllocCPUS
```

## Files / commits

| Commit | What |
|---|---|
| `406d3c2` | Local `main` fast-forwarded to PR #5 (perf) |
| `1cb50e6` | General aligners (rebased from `bca13f4`) |
| `70d4f83` | `docs/candidate_datasets.md`; 31-movie comparison in `docs/general_registration.md` |
| `b1943b6` | Merge `feature/general-aligners` |
| `3f5c2bc` | SLURM support: `run_optomerge.py`, `grouping.py`, `_kernels/{_parallel,io,transform,processing}.py`, `config.py`, `__init__.py`, `hpc/*`, `tests/test_sharding.py`, `tests/test_io.py`, README |
| `b651766` | Merge `feature/slurm-hpc` |
| (this commit) | `journals/` (backfill + this entry), `CLAUDE.md` |

## Open items / next steps

- **Push** `main` and the two feature branches from the Mac (none of this is on
  GitHub yet).
- **Pilot on the cluster** with `optomerge_single.sbatch` on the 31 test movies:
  - confirm the environment
  - measure the real `MEM_PER_MOVIE_GB` and the time per movie
  - spot-check feature-aligner overlays
  - then size `--array`
- **Redo with other aligners** by pointing `output` at a new folder (for example
  `aligned_mi`).
- **Datasets with different channel orders** currently need one array per config.
  A FASTrack-style manifest is possible future work.
- **FASTrack's own `hpc/` scripts** locate `env.sh` via `$(dirname "$(realpath
  "$0")")`, which under sbatch points into SLURM's spool directory. Check whether
  they work on the cluster; optomerge's `_common.sh` uses `$SLURM_SUBMIT_DIR`
  instead.
- Carried over:
  - `--channel-order auto` is unreliable
  - sidecar save/load of a fitted alignment
  - tighter transform padding (needs a fidelity sign-off)
  - GPU support
  - MI vs feature visual comparison
  - Deep-LASI validation
  - mark `SPEEDUP_PROPOSAL.md` stale
  - decide on `docs/skills/`
