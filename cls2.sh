python run_classification.py \
--model="gpt2-medium" \
--dataset="sst2" \
--num_seeds=1 \
--start_seed=0 \
--all_shots="0" \
--bs=32 \
--method='gmm_test_estimate' 