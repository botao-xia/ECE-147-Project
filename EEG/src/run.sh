#!/bin/bash

GPU=0
SEED=42
CKPT_NAME="EEGNet_mod_person_1"
CKPT_DIR="./outputs/EEGNet_mod"
DATA_DIR="./EEG_Dataset/"
MODEL_NAME="EEGNet_mod"

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
        --accumulate_grad_batches 1 \
        --timestep_start 400 \
        --timestep_end 1000 \

#         --train_person_index 0 1 2 3 4 5 \
#         --test_person_index 6 7 8 \