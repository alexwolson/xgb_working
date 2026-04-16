#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000M
#SBATCH --time=1-0:0:0
#SBATCH --output=%N-%j.out

module load python/3.10

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirementscomputecan.txt

uv run predict --config-file config/training_config_NewFeaturesTest_20250625_try3.json \
--data-directory data