#!/bin/bash

LR=0.00001
MODEL="roberta-base"
DO_LOWER_CASE=False
DATASET_TO_TRAIN_ON="ACE2005"
DATASET_TO_FEW_SHOT_EVALUATE_ON="MAVEN"
DATASET_TRAIN_PATH="../../data/raw/ace/train.json"
DATASET_VALID_PATH="../../data/raw/ace/dev.json"
DATASET_TEST_PATH="../../data/raw/ace/test.json"
DATASET_TRAIN_FEW_SHOT_PATH="../../data/raw/maven/train.jsonl"
DATASET_TEST_TO_EVALUATE_ON_PATH="../../data/raw/maven/valid.jsonl"
EPOCHS=10
MULTIPLICATIVE_LR=0.99
BATCH_SIZE=27
BATCH_SIZE_FEW_SHOT=5
AVERAGE_OVER_RUNS_NUM=5
FEW_SHOT_EXAMPLES_NUM=500
TRAINING_TASK="trigger identification"
SEED=$1
NAME="roberta_ti_pretrained_ace_joint_train_on_ace_500_shot_predict_on_maven_average_over_5_runs_SEED=${SEED}"
DEVICE="cuda:0"
LOAD_PRETRAINED_MODEL_FROM="../../models_seed_${SEED}/ace_trigger_model"

python ../../baselines/ti_joint_training_few_shot_target_evaluation.py \
	--lr $LR \
	--model "$MODEL" \
	--do_lower_case $DO_LOWER_CASE \
	--dataset_to_train_on "$DATASET_TO_TRAIN_ON" \
	--dataset_to_few_shot_evaluate_on "$DATASET_TO_FEW_SHOT_EVALUATE_ON" \
	--dataset_train_path "$DATASET_TRAIN_PATH" \
	--dataset_valid_path "$DATASET_VALID_PATH" \
	--dataset_test_path "$DATASET_TEST_PATH" \
	--dataset_train_few_shot_path "$DATASET_TRAIN_FEW_SHOT_PATH" \
	--dataset_test_to_evaluate_on_path "$DATASET_TEST_TO_EVALUATE_ON_PATH" \
	--epochs $EPOCHS \
	--multiplicative_lr $MULTIPLICATIVE_LR \
    --batch_size $BATCH_SIZE \
	--batch_size_few_shot $BATCH_SIZE_FEW_SHOT \
	--average_over_runs_num $AVERAGE_OVER_RUNS_NUM \
	--few_shot_examples_num $FEW_SHOT_EXAMPLES_NUM \
	--training_task "$TRAINING_TASK" \
	--name "$NAME" \
	--seed $SEED \
	--device "$DEVICE" \
	--load_pretrained_model_from "$LOAD_PRETRAINED_MODEL_FROM"