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

python run_study.py --targets Count_EX1 Count_EX2 \
--study-name ComputeCan_20250625_test1 \
--study_count 10 \
--onehot-encoding \
--sen-geometrical \
--discard-features "time[s]" \
--data-directory data \
--subsample-shap \
--multithread \
--evaluate-only \
--tree-method hist