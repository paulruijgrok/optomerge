#!/bin/bash
#SBATCH --job-name=optomerge_compare
#SBATCH --output=/scratch/users/ruijgrok/optomerge_runs/MyLOV1Char/slurm_compare_%j.out
#SBATCH --error=/scratch/users/ruijgrok/optomerge_runs/MyLOV1Char/slurm_compare_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --partition=possu
#
# Run the MyLOV1Char pilot with other aligners and diagnose each run, for a
# side-by-side with the feature aligner (aligned_feature_v2). Same configs as
# run.sbatch; only the method (and output folder) change:
#   aligned_<method>/<date>/                      merged movies + run logs
#   diagnostics_<method>/summary.{csv,png}, PNGs  diagnose_alignment.py output
#
# Usage (from the repo root):
#   sbatch hpc/runs/MyLOV1Char/compare_aligners.sbatch              # ecc mi
#   METHODS="phase ecc" sbatch hpc/runs/MyLOV1Char/compare_aligners.sbatch
set -euo pipefail

_REPO="${OPTOMERGE_REPO:-${SLURM_SUBMIT_DIR:-.}}"
if [ ! -f "$_REPO/hpc/_common.sh" ]; then
    echo "ERROR: $_REPO is not the optomerge repo. Submit from the repo root," >&2
    echo "       or set OPTOMERGE_REPO=/path/to/optomerge." >&2
    exit 2
fi
WORKERS="${WORKERS:-6}"
source "$_REPO/hpc/_common.sh"
RUN="$REPO/hpc/runs/MyLOV1Char"
ROOT=/scratch/users/ruijgrok/optomerge_runs/MyLOV1Char
METHODS="${METHODS:-ecc mi}"

echo "host=$(hostname)  cpus=${CPUS}  workers=${WORKERS}  methods=${METHODS}  repo=$REPO @ $(cd "$REPO" && git rev-parse --short HEAD)"

rc=0
for method in $METHODS; do
    for date in 20170803 20180620 20181002; do
        echo "=== $method $date  $(date +%T)"
        python "$REPO/run_optomerge.py" --config "$RUN/base.toml" "$RUN/$date.toml" \
            --align-method "$method" --output "$ROOT/aligned_$method/$date" \
            --workers "$WORKERS" "$@" || { echo "=== $method $date: FAILED"; rc=1; }
    done
    echo "=== diagnose $method  $(date +%T)"
    python "$RUN/diagnose_alignment.py" "$ROOT/diagnostics_$method" --workers "$CPUS" \
        --aligned "aligned_$method" || { echo "=== diagnose $method: FAILED"; rc=1; }
done
echo "=== done $(date +%T)  rc=$rc"
exit $rc
