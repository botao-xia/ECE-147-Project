#!/bin/bash

GPU=0
SEED=42
CKPT_NAME="EEGNet"
CKPT_DIR="./outputs/EEGNet_Ckpts"
DATA_DIR="/Users/shawn/Desktop/BX/ECE147/ECE-147-Project/EEG_Dataset/"

python main.py \
        --data_dir=${DATA_DIR} \
        --model_name="EEGNet" \
        --ckpt_name=${CKPT_NAME} \
        --ckpt_dir=${CKPT_DIR} \
        --random_state=${SEED} \
        --train_epochs 20 \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --accumulate_grad_batches 1 \