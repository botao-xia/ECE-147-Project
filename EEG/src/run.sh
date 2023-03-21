#!/bin/bash

GPU=0
SEED=42
CKPT_NAME="Modified_EEGNet_trim"
CKPT_DIR="./outputs/Modified_EEGNet_trim"
DATA_DIR="./EEG_Dataset/"
MODEL_NAME="EEGNet_Modified"

python main.py \
        --data_dir=${DATA_DIR} \
        --ckpt_name=${CKPT_NAME} \
        --ckpt_dir=${CKPT_DIR} \
        --random_state=${SEED} \
        --model_name=${MODEL_NAME} \
        --gpus="${GPU}" \
        --train_epochs 200 \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --timestep_start 0 \
        --timestep_end 600 \
        --accumulate_grad_batches 1 \
