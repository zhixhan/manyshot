python finetune.py \
--model="gpt2-xl" \
--dataset="sst2" \
--bs=8 \
--epochs=20 \
--lr=1e-5 \
--seed=1 \
--num_shot=32 \
--output_dir='/home/v-zhixhan/data/zhixhan/many_shot_logs/gpt2-xl-rte-demons-manyshot-ft' \
--evaluate_only=''

