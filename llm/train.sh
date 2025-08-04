#!/bin/bash

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_IBEXT_DISABLE=1
# export PDSH_RCMD_TYPE=ssh

theta0=Qwen/Qwen2.5-0.5B-Instruct
theta0_str=$(echo ${theta0}|sed 's/\//-/g')

exp_note="Daunce-demo"
exp_config=config/qwen-math-demo.yaml

k=100
micro_batch_size=8

log_file=logs/${exp_note}.log

gamma_search=(1)
lr_search=(3e-4)
rho_search=(1)

for gamma in ${gamma_search[@]}; do
    for lr in ${lr_search[@]}; do
        for rho in ${rho_search[@]}; do
            
            echo "gamma: $gamma, lr: $lr, rho: $rho"
            exp_id=${exp_note}-k${k}-theta0${theta0_str}_roh${rho}_lr${lr}_gamma${gamma}
            echo "exp_id: $exp_id"
            output_dir=./checkpoints/checkpoint-${exp_note}/$exp_id/
            mkdir -p $output_dir
            
            for i in $(seq 1 $k)
            do
                # count the number of checkpoints
                ckpt_count=$(ls -l $output_dir | grep -c ^d)
                if [ $ckpt_count -ge $k ]; then
                    echo "Checkpoint count $ckpt_count reached the limit $k"
                    break
                fi
                
                seed=$((RANDOM))
                ckpt_dir=${output_dir}/model_seed-${seed}/
                if [ -f ${ckpt_dir} ]; then
                    seed=$((RANDOM))
                    ckpt_dir=${output_dir}/model_seed-${seed}/
                fi
                mkdir -p $ckpt_dir

                accelerate launch --config_file config/fsdp_config_qwen.yaml daunce-if.py \
                    --config $exp_config \
                    --rho $rho \
                    --lr $lr \
                    --gamma $gamma \
                    --wandb_run_name $exp_id-seed${seed} \
                    --save_dir $ckpt_dir \
                    --pseudo_random $seed 
            done
        done
    done
done



for gamma in ${gamma_search[@]}; do
    for lr in ${lr_search[@]}; do
        for rho in ${rho_search[@]}; do
            exp_id=${exp_note}-k${k}-theta0${theta0_str}_roh${rho}_lr${lr}_gamma${gamma}
            echo "exp_id: $exp_id"
            output_dir=./checkpoints/checkpoint-${exp_note}/$exp_id/

            python3 model_output.py \
                --exp_id $exp_id \
                --config ${exp_config} \
                --model-ckpt $output_dir 
        done
    done
done