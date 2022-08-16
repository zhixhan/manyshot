python run_classification.py \
--model="gpt2-xl" \
--dataset="sst2" \
--num_seeds=5 \
--start_seed=2 \
--all_shots="1" \
--bs=2 \
--method='ori' \
--subsample_test_set=300