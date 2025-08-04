#!/bin/bash

accelerate launch --config_file config/fsdp_config_single_qwen.yaml gen_embedding.py \
    --config config/qwen-math-demo.yaml
