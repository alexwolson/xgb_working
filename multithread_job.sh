sbatch -W multithread_study_train.sh
wait

sbatch -W multithread_study_eval.sh
wait