#!/bin/bash

LR=0.00001
LR_EVENT_EMB=0.00005
MODEL="roberta-base"
DO_LOWER_CASE=False
DATASET_TO_TRAIN_ON="ACE2005"
DATASET_TRAIN_PATH="../data/raw/ace/train.json"
DATASET_VALID_PATH="../data/raw/ace/dev.json"
DATASET_TEST_PATH="../data/raw/ace/test.json"
DATASET_TRAIN_MINI_EXTRACTIONS_PATH="../data/processed/mini/ace_train_triplets_filtered_merged.json"
DATASET_VALID_MINI_EXTRACTIONS_PATH="../data/processed/mini/ace_dev_triplets_filtered_merged.json"
DATASET_TEST_MINI_EXTRACTIONS_PATH="../data/processed/mini/ace_test_triplets_filtered_merged.json"
EPOCHS=10
MULTIPLICATIVE_LR=0.99
BATCH_SIZE=32
TRAINING_TASK="trigger identification"
SEED=$1
NAME="roberta_ti_train_on_ace_implicit_multitask_save_learned_model_SEED=${SEED}"
LABEL_HIDDEN_SIZE=300
DEVICE="cuda:0"
SAVE_TO="../models_seed_${SEED}/ace_implicit_multitask_model"

python ../baselines/ti_rl_implicit_multitask_training.py \
	--lr $LR \
	--lr_event_emb $LR_EVENT_EMB \
	--model "$MODEL" \
	--do_lower_case $DO_LOWER_CASE \
	--dataset_to_train_on "$DATASET_TO_TRAIN_ON" \
	--dataset_train_path "$DATASET_TRAIN_PATH" \
	--dataset_valid_path "$DATASET_VALID_PATH" \
	--dataset_test_path "$DATASET_TEST_PATH" \
	--dataset_train_mini_extractions_path "$DATASET_TRAIN_MINI_EXTRACTIONS_PATH" \
	--dataset_valid_mini_extractions_path "$DATASET_VALID_MINI_EXTRACTIONS_PATH" \
	--dataset_test_mini_extractions_path "$DATASET_TEST_MINI_EXTRACTIONS_PATH" \
	--epochs $EPOCHS \
	--multiplicative_lr $MULTIPLICATIVE_LR \
	--batch_size $BATCH_SIZE \
	--training_task "$TRAINING_TASK" \
	--name "$NAME" \
	--seed $SEED \
	--label_hidden_size $LABEL_HIDDEN_SIZE \
	--device "$DEVICE" \
	--save_to "$SAVE_TO"