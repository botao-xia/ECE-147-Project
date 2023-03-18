#!/bin/bash

GPU=0
SEED=42
CKPT_NAME="subject_test_model"
CKPT_DIR="/Users/shawn/Desktop/BX/ECE147/ECE-147-Project/EEG/src/outputs/subject"
DATA_DIR="./EEG_Dataset/"
MODEL_NAME="EEGNet_mod"

python main.py \
        --data_dir=${DATA_DIR} \
        --ckpt_name=${CKPT_NAME} \
        --ckpt_dir=${CKPT_DIR} \
        --random_state=${SEED} \
        --model_name=${MODEL_NAME} \
        --gpus="${GPU}" \
        --train_epochs 20 \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --accumulate_grad_batches 1 \
        --train_person_index 0 1 2 3 4 5 6 7 8 \
        --test_person_index 0 \
