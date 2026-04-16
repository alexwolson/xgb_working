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
pip install --no-index -r requirementscomputecan.txt

srun uv run tune --targets Count_EX1 Count_EX2 \
--study-name ComputeCan_20250625_test1 \
--study-count 10 \
--onehot-encoding \
--sen-geometrical \
--discard-features "time[s]" \
--data-directory data/raw/Organized_Data \
--subsample-shap \
--multithread \
--tree-method hist