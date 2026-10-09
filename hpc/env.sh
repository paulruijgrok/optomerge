#!/bin/bash
# Sourced by the optomerge SLURM scripts to activate the Python environment on a
# compute node. Edit CONDA_ENV (and add any `module load` lines your cluster
# needs) once here, and every script picks it up.
#
# Same conda-activation idiom as FASTrack's hpc/env.sh:
#   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate <env>

CONDA_ENV="${CONDA_ENV:-optomerge}"

# module load python/3.11   # uncomment if your cluster provides conda via a module
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
