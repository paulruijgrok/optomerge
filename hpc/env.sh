#!/bin/bash
# Sourced by the optomerge SLURM scripts to activate the Python environment on a
# compute node. Edit CONDA_ENV (and add any `module load` lines your cluster
# needs) once here, and every script picks it up.
#
# Same conda-activation idiom as FASTrack's hpc/env.sh:
#   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate <env>
#
# CONDA_ENV may be an env name or a prefix path. CONDA_BASE overrides which conda
# install provides the activation script (default: whatever `conda` is on PATH).
#
# Sherlock (Stanford): the env lives in $GROUP_HOME (never $SCRATCH, which is
# purged) and is built with the group's own miniconda; see hpc/README.md.

if [ -n "${GROUP_HOME:-}" ] && [ -d "$GROUP_HOME/envs/optomerge" ]; then
    CONDA_ENV="${CONDA_ENV:-$GROUP_HOME/envs/optomerge}"
    CONDA_BASE="${CONDA_BASE:-$GROUP_HOME/miniconda}"
fi
CONDA_ENV="${CONDA_ENV:-optomerge}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"

# module load python/3.11   # uncomment if your cluster provides conda via a module
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "python=$(command -v python)  env=$CONDA_ENV"
