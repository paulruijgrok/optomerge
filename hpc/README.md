# Running optomerge on a SLURM cluster

Thin `sbatch` wrappers around `run_optomerge.py`, modelled on FASTrack's `hpc/`
scripts (same conda idiom, `--partition=possu`, `--requeue`). You normally edit
only the resource lines, `env.sh`, and a run config TOML.

optomerge parallelises at two levels:

- **Across nodes**: a job array splits the movie list into shards
  (`--num-shards` / `--shard-index`). Every task sees the same sorted listing and
  keeps its own round-robin share, so tasks need no coordination and all write
  into one output tree.
- **Within a node**: `--workers N` runs N movies at once, each capped to its share
  of the allocated cores. Throughput is **memory-bound**
  ([`docs/benchmarks.md`](../docs/benchmarks.md)), so `hpc/_common.sh` picks
  `WORKERS = min(cpus, mem_GB / MEM_PER_MOVIE_GB)` (default 5 GB per movie).

Thread pools are sized from the CPUs actually granted to the job (CPU affinity /
`SLURM_CPUS_PER_TASK`), not from every core on the node.

## One-time setup

1. Clone and install into a conda env on the cluster (login node is fine):
   ```bash
   git clone https://github.com/paulruijgrok/optomerge.git && cd optomerge
   conda create -n optomerge python=3.11 -y && conda activate optomerge
   pip install -e .              # OpenCV (headless) comes in as a core dependency
   pytest -q                     # optional sanity check
   ```
2. Edit `hpc/env.sh` if your env isn't called `optomerge` (or export
   `CONDA_ENV=...`), and add any `module load` lines your cluster needs.
3. In each `.sbatch`, set `--partition` / `--account` and, optionally,
   `--mail-user`.
4. Copy `hpc/example_cluster.toml` to e.g. `my_run.toml` and set **absolute**
   `input` / `output` paths and the dataset's `channel_order`.

### Sherlock (Stanford)

The CentOS 7 login and compute nodes have a glibc too old for current scipy /
OpenCV wheels, so `pip install -e .` tries to compile scipy and fails (no
OpenBLAS). Take the compiled dependencies from conda-forge and install optomerge
itself with `--no-deps`. The group's miniconda has the libmamba solver; the old
classic solver in the possu base install OOMs at 16 GB on this solve. Build inside
a job, not on a login node:

```bash
srun -p dev -c 4 --mem=16G -t 00:30:00 bash -c '
  export CONDA_PKGS_DIRS=$SCRATCH/cache/conda_pkgs
  $GROUP_HOME/miniconda/bin/conda create -y -p $GROUP_HOME/envs/optomerge \
      -c conda-forge --override-channels python=3.11 numpy scipy scikit-image \
      tifffile opencv pytest matplotlib
  source $GROUP_HOME/miniconda/etc/profile.d/conda.sh
  conda activate $GROUP_HOME/envs/optomerge
  pip install --no-deps -e . && pytest -q'
```

`hpc/env.sh` picks up `$GROUP_HOME/envs/optomerge` automatically. Keep the env in
`$GROUP_HOME`, never `$SCRATCH` (purged after 90 days untouched). Inputs and
outputs go on `$SCRATCH`. A TOML can't expand variables, so write the expanded
path (`/scratch/users/<sunetid>/...`). `--partition=possu` is the group partition;
`sh_part` lists the others. `pip check` will report `opencv-python-headless`
missing because conda's `opencv` provides `cv2` under a different distribution
name. This is harmless.

Per-dataset run configs and their job scripts live in `hpc/runs/<dataset>/`
(e.g. `hpc/runs/MyLOV1Char/`: a shared `base.toml` layered under one TOML per
recording date, because the channel order differs between dates).

Submit **from the repo root**. sbatch runs a spool copy of the script, so the
scripts find the repo through the submission directory; to submit from elsewhere,
`export OPTOMERGE_REPO=/path/to/optomerge`.

## Pilot: one node

Run the 31-movie test set (or any small set) as a single job first. It checks the
environment and gives you time and memory per movie for sizing the array.

```bash
sbatch hpc/optomerge_single.sbatch my_run.toml
# afterwards: peak memory and wall time
sacct -j <jobid> -o JobID,Elapsed,MaxRSS,AllocCPUS
```

If `MaxRSS / WORKERS` differs a lot from 5 GB, set `MEM_PER_MOVIE_GB` (or
`WORKERS`) for the big run.

## Many movies: job array

```bash
python run_optomerge.py --config my_run.toml --dry-run | head -1   # N movies
# choose T tasks so each finishes well inside --time, then edit
#   #SBATCH --array=0-(T-1)   in hpc/optomerge_array.sbatch
sbatch hpc/optomerge_array.sbatch my_run.toml
```

Each task logs to `<output>/run_log_shardNNNofMMM.txt`; SLURM's own stdout and
stderr go to `optomerge_<job>_<task>.out/.err` in the submission directory. A
shard that gets no files (T larger than N) exits cleanly.

Sharding is round-robin over **movies** with `reuse_alignment = "none"` or a
reference FILE. With `reuse_alignment = "first"` it is round-robin over whole
**alignment sets**, because each set's first movie seeds the rest. In that case,
group with `--group-by token` or `folder` so there are enough sets to spread.

## Restart / resume

Outputs are written to a hidden `.partial` file and renamed when complete, and
existing outputs are skipped. A preempted, timed-out, or re-submitted task
therefore resumes without redoing finished movies and never treats a truncated
file as done. Just `sbatch` the same command again. Pass `--overwrite` to
recompute.

## Redoing a run with different settings

Runs are cheap to repeat. To try another aligner, point `output` at a new folder
(or override flags on the command line) and resubmit:

```bash
sbatch hpc/optomerge_array.sbatch my_run.toml --mi                  # same output dir: skipped!
sbatch hpc/optomerge_array.sbatch my_run_mi.toml                    # output = .../aligned_mi
sbatch hpc/optomerge_array.sbatch my_run.toml --mi --overwrite      # replace in place
```

Each output folder keeps a `run_config.toml` with the exact settings used.

## Quick checklist before a big array

```bash
python run_optomerge.py --config my_run.toml --dry-run | head -1    # N movies, right input?
python run_optomerge.py --config my_run.toml --dry-run --num-shards 4 --shard-index 0 | head -1
sbatch hpc/optomerge_single.sbatch my_run.toml                      # pilot (small set)
sbatch hpc/optomerge_array.sbatch my_run.toml                       # full run
```
