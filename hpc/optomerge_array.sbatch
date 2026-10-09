#!/bin/bash
#SBATCH --job-name=optomerge
#SBATCH --output=optomerge_%A_%a.out
#SBATCH --error=optomerge_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=possu
##SBATCH --account=<your_account>      # uncomment + set if your cluster requires an allocation
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@stanford.edu  # uncomment + set to get notified
#SBATCH --requeue
#SBATCH --array=0-3                     # T tasks (indices 0..T-1); see sizing note below
#
# Run optomerge over a whole input tree as a job array. Every task runs the same
# command on its own shard of the (sorted) movie list -- round-robin over movies,
# or over whole alignment sets with --reuse-alignment first -- and all tasks write
# into the same output tree (one run_log_shardNNNofMMM.txt each). Within a task,
# WORKERS movies run in parallel (see hpc/_common.sh).
#
# Restartable: outputs are written atomically and existing outputs are skipped,
# so a requeued, timed-out, or re-submitted task picks up where it left off.
#
# SIZING -- pick T = number of array tasks:
#   python run_optomerge.py --config my_run.toml --dry-run | head -1   # N movies
#   Each task handles ~N/T movies, WORKERS at a time. Choose T so a task finishes
#   well inside --time (budget ~1 min/movie/worker to start; refine from the pilot),
#   then edit "#SBATCH --array=0-$((T-1))" above. NUM_SHARDS follows the array range.
#
# Usage (from the repo root):
#   sbatch hpc/optomerge_array.sbatch my_run.toml
#   sbatch hpc/optomerge_array.sbatch my_run.toml --feature --fit-rotation
#   WORKERS=8 sbatch hpc/optomerge_array.sbatch my_run.toml
# The TOML holds input/output/channel order/method (see hpc/example_cluster.toml);
# extra arguments are passed straight to run_optomerge.py and override it.
set -euo pipefail

CONFIG="${1:?usage: sbatch optomerge_array.sbatch <config.toml> [extra run_optomerge args]}"
shift || true
CONFIG="$(realpath "$CONFIG")"

# sbatch runs a spool copy of this script, so find the repo via the submit dir.
_REPO="${OPTOMERGE_REPO:-${SLURM_SUBMIT_DIR:-.}}"
if [ ! -f "$_REPO/hpc/_common.sh" ]; then
    echo "ERROR: $_REPO is not the optomerge repo. Submit from the repo root," >&2
    echo "       or set OPTOMERGE_REPO=/path/to/optomerge." >&2
    exit 2
fi
source "$_REPO/hpc/_common.sh"

NUM_SHARDS="${NUM_SHARDS:-$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))}"
SHARD_INDEX=$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))

echo "host=$(hostname)  shard=${SHARD_INDEX}/${NUM_SHARDS}  cpus=${CPUS}  workers=${WORKERS}  config=$CONFIG  repo=$REPO"
python "$REPO/run_optomerge.py" \
    --config "$CONFIG" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --workers "$WORKERS" \
    "$@"
