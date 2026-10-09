#!/bin/bash
#SBATCH --job-name=optomerge
#SBATCH --output=optomerge_%j.out
#SBATCH --error=optomerge_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=possu
##SBATCH --account=<your_account>      # uncomment + set if your cluster requires an allocation
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@stanford.edu  # uncomment + set to get notified
#SBATCH --requeue
#
# Run optomerge over an input tree as ONE job on one node (WORKERS movies at a
# time). Use it for a pilot run -- e.g. the 31-movie test_data set -- to measure
# time and memory per movie before sizing a job array.
#
# Usage (from the repo root):
#   sbatch hpc/optomerge_single.sbatch my_run.toml
#   sbatch hpc/optomerge_single.sbatch my_run.toml --feature --fit-rotation
#   sbatch --cpus-per-task=32 --mem=128G hpc/optomerge_single.sbatch my_run.toml
set -euo pipefail

CONFIG="${1:?usage: sbatch optomerge_single.sbatch <config.toml> [extra run_optomerge args]}"
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

echo "host=$(hostname)  cpus=${CPUS}  workers=${WORKERS}  config=$CONFIG  repo=$REPO"
python "$REPO/run_optomerge.py" --config "$CONFIG" --workers "$WORKERS" "$@"
