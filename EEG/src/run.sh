#!/bin/bash

GPU=0
SEED=42
CKPT_NAME="EEGNet_mod"
CKPT_DIR="./outputs/EEGNet_mod"
DATA_DIR="./EEG_Dataset/"
MODEL_NAME="EEGNet_mod"

python main.py \
        --data_dir=${DATA_DIR} \
        --model_name="EEGNet" \
        --ckpt_name=${CKPT_NAME} \
        --ckpt_dir=${CKPT_DIR} \
        --random_state=${SEED} \
        --model_name=${MODEL_NAME} \
        --gpus="${GPU}" \
        --train_epochs 10 \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --accumulate_grad_batches 1 \