#!/bin/bash
# Shared setup for the optomerge sbatch scripts (sourced, not run).
#
# Locates the repo, activates the env, and picks --workers from the job's memory.
# Note: sbatch runs a *copy* of the script from SLURM's spool directory, so the
# script cannot find the repo via its own path ($0). We use OPTOMERGE_REPO if set,
# else the submission directory -- so submit from the repo root, or export
# OPTOMERGE_REPO=/path/to/optomerge.

REPO="${OPTOMERGE_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "$REPO/run_optomerge.py" ]; then
    echo "ERROR: run_optomerge.py not found in REPO=$REPO." >&2
    echo "       Submit from the optomerge repo root, or set OPTOMERGE_REPO." >&2
    exit 2
fi
source "$REPO/hpc/env.sh"

CPUS="${SLURM_CPUS_PER_TASK:-1}"

# Movies processed concurrently on this node. Throughput is memory-bound (see
# docs/benchmarks.md), so size it from the job's RAM, not just its cores:
#   workers = min(cpus, mem_GB / MEM_PER_MOVIE_GB), at least 1.
# MEM_PER_MOVIE_GB defaults to 5 (float32 merge of a 512x512x800 movie peaks at
# roughly 3-4 GB); measure yours with `sacct -j <job> -o MaxRSS` after a pilot.
# Override directly with WORKERS=<n>.
MEM_PER_MOVIE_GB="${MEM_PER_MOVIE_GB:-5}"
if [ -z "${WORKERS:-}" ]; then
    if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
        MEM_GB=$(( SLURM_MEM_PER_NODE / 1024 ))           # SLURM reports MB
    elif [ -n "${SLURM_MEM_PER_CPU:-}" ]; then
        MEM_GB=$(( SLURM_MEM_PER_CPU * CPUS / 1024 ))
    else
        MEM_GB=$(( CPUS * MEM_PER_MOVIE_GB ))              # unknown: assume enough
    fi
    WORKERS=$(( MEM_GB / MEM_PER_MOVIE_GB ))
    # (if-blocks, not `[ ] && x`: a false test would trip `set -e` in the caller)
    if [ "$WORKERS" -gt "$CPUS" ]; then WORKERS=$CPUS; fi
    if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
fi
