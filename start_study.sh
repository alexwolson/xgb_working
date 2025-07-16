python run_study.py --targets Count_EX1 \
--study-name lagtest \
--study_count 3 \
--onehot-encoding \
--sen-geometrical \
--discard-features "time[s]" \
--data-directory Data \
--subsample-shap \
--feature-lag-amount 3 \
--lagged-features "AN_1_LL[m/s],AN_2_LQ[m/s],AN_3_RQ[m/s],AN_4_RR[m/s]"