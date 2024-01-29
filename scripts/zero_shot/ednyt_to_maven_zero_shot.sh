#!/bin/bash

LR=0.00001
MODEL="roberta-base"
DO_LOWER_CASE=False
DATASET_TO_TRAIN_ON="EDNYT"
DATASET_TO_ZERO_SHOT_EVALUATE_ON="MAVEN"
DATASET_TRAIN_PATH="../../data/processed/ednyt/train.json"
DATASET_VALID_PATH="../../data/processed/ednyt/valid.json"
DATASET_TEST_PATH="../../data/processed/ednyt/test.json"
DATASET_TEST_TO_EVALUATE_ON_PATH="../../data/raw/maven/valid.jsonl"
EPOCHS=10
MULTIPLICATIVE_LR=0.99
BATCH_SIZE=32
TRAINING_TASK="trigger identification"
SEED=$1
NAME="roberta_ti_train_on_ednyt_zero_shot_predict_on_maven_SEED=${SEED}"
DEVICE="cuda:0"
SAVE_TO="../../models_seed_${SEED}/ednyt_trigger_model"

python ../../baselines/ti_training_zero_shot_target_evaluation.py \
	--lr $LR \
	--model "$MODEL" \
	--do_lower_case $DO_LOWER_CASE \
	--dataset_to_train_on "$DATASET_TO_TRAIN_ON" \
	--dataset_to_zero_shot_evaluate_on "$DATASET_TO_ZERO_SHOT_EVALUATE_ON" \
	--dataset_train_path "$DATASET_TRAIN_PATH" \
	--dataset_valid_path "$DATASET_VALID_PATH" \
	--dataset_test_path "$DATASET_TEST_PATH" \
	--dataset_test_to_evaluate_on_path "$DATASET_TEST_TO_EVALUATE_ON_PATH" \
	--epochs $EPOCHS \
	--multiplicative_lr $MULTIPLICATIVE_LR \
    --batch_size $BATCH_SIZE \
	--training_task "$TRAINING_TASK" \
	--name "$NAME" \
	--seed $SEED \
	--device "$DEVICE" \
	--save_to "$SAVE_TO"