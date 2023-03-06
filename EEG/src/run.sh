#!/bin/bash

GPU=0
SEED=42
CKPT_NAME="EEG_shallowCNN"
CKPT_DIR="./outputs/EEG_shallowCNN_Ckpts"
DATA_DIR="./EEG_Dataset/"

python src/main.py \
        --data_dir=${DATA_DIR} \
        --ckpt_name=${CKPT_NAME} \
        --ckpt_dir=${CKPT_DIR} \
        --random_state=${SEED} \
        --gpus="${GPU}" \
        --train_epochs 10 \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --accumulate_grad_batches 1 \
