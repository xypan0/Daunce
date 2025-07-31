#!/bin/bash
set -e

output_dir="<your output directory here>"
theta_0_model="<path to your theta_0 model here>"
exp_id="daunce-cifar-resnet"

log_file="${output_dir}/run.log"
k=3 # default number of models
epoch=1 # default number of epochs


for i in $(seq 1 $k)
do
    seed=$((RANDOM))
    ckpt_dir=${output_dir}/model-seed-${seed}/
    mkdir -p $ckpt_dir


    python3 ./daunce-efim.py \
        --model-ckpt ${theta_0_model} \
        --lr 3e-2 \
        --rho 1.0 \
        --save $ckpt_dir \
        --epoch ${epoch} \
        --save_interval ${epoch} \
        --gamma 1.0 \
        --ratio 0.3 \
        --pseudo_random $seed >> $log_file
done


# generate model output (e.g. margin, loss, logits, etc.)
python3 ./model_output.py \
    --model-ckpt ${output_dir} \
    --epoch ${epoch} \
    --exp-id ${exp_id} >> $log_file

# visualize results with visualize.ipynb