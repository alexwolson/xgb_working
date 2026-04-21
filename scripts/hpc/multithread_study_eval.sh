#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3G
#SBATCH --ntasks=10
#SBATCH --time=1-0:0:0
#SBATCH --output=%N-%j.out

module load python/3.10

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements-computecan.txt

srun uv run tune config/base.toml
