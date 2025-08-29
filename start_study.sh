uv run run_study.py --targets Count_EX1 Count_EX2 \
--study-name cftest0812-3 \
--study_count 100 \
--onehot-encoding \
--sen-geometrical \
--discard-features "time[s]" \
--data-directory data \
--subsample-shap \
--tree-method gpu_hist \
--lagged-features "AN_1_LL[m/s],AN_2_LQ[m/s],AN_3_RQ[m/s],AN_4_RR[m/s],ML_LL[mm],ML_LQ[mm],ML_RQ[mm],ML_RR[mm]" \
--feature-lag-amount 10 \
--mould-position \
--clogging-factors 

uv run run_study.py --targets Count_EX1 Count_EX2 \
--study-name data_cleaned_only_model \
--study_count 100 \
--onehot-encoding \
--data-directory data \
--subsample-shap \
--tree-method gpu_hist \
--clogging-factors