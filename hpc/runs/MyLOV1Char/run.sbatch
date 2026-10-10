#!/bin/bash
#SBATCH --job-name=optomerge_mylov1char
#SBATCH --output=/scratch/users/ruijgrok/optomerge_runs/MyLOV1Char/slurm_%j.out
#SBATCH --error=/scratch/users/ruijgrok/optomerge_runs/MyLOV1Char/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --partition=possu
#
# MyLOV1Char pilot: the three recording dates in sequence on one node, WORKERS
# movies at a time within each date (each capped to ~CPUS/WORKERS threads; see
# hpc/_common.sh). A date whose run has failures doesn't stop the others; the job
# exits non-zero if any date failed. Re-submitting resumes (finished outputs skip).
#
# Usage (from the repo root):
#   sbatch hpc/runs/MyLOV1Char/run.sbatch
#   WORKERS=16 sbatch hpc/runs/MyLOV1Char/run.sbatch          # N movies at once
#   sbatch hpc/runs/MyLOV1Char/run.sbatch --overwrite         # extra args -> run_optomerge.py
set -euo pipefail

_REPO="${OPTOMERGE_REPO:-${SLURM_SUBMIT_DIR:-.}}"
if [ ! -f "$_REPO/hpc/_common.sh" ]; then
    echo "ERROR: $_REPO is not the optomerge repo. Submit from the repo root," >&2
    echo "       or set OPTOMERGE_REPO=/path/to/optomerge." >&2
    exit 2
fi
# 8 movies at once x 4 threads each on 32 CPUs (the largest date has 14 movies).
# Without this, _common.sh sizes from memory: min(32, 128/5) = 25 single-thread
# workers, which wastes cores on sets of 6-14 movies.
WORKERS="${WORKERS:-8}"
source "$_REPO/hpc/_common.sh"
RUN="$REPO/hpc/runs/MyLOV1Char"

echo "host=$(hostname)  cpus=${CPUS}  workers=${WORKERS}  repo=$REPO @ $(cd "$REPO" && git rev-parse --short HEAD)"
python -c "import numpy, scipy, cv2, skimage; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'cv2', cv2.__version__, 'skimage', skimage.__version__)"

rc=0
for date in 20170803 20180620 20181002; do
    echo "=== $date  $(date +%T)"
    python "$REPO/run_optomerge.py" --config "$RUN/base.toml" "$RUN/$date.toml" \
        --workers "$WORKERS" "$@" || { echo "=== $date: FAILED (exit $?)"; rc=1; }
done
echo "=== done $(date +%T)  rc=$rc"
exit $rc
