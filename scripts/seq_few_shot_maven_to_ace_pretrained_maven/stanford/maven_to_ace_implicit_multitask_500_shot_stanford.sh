#!/bin/bash

LR=0.00001
MODEL="roberta-base"
DO_LOWER_CASE=False
DATASET_THE_MODEL_WAS_PRETRAINED_ON="MAVEN"
DATASET_TO_TRAIN_AND_FEW_SHOT_EVALUATE_ON="ACE2005"
DATASET_TRAIN_FEW_SHOT_PATH="../../../data/raw/ace/train.json"
DATASET_TEST_FEW_SHOT_PATH="../../../data/raw/ace/test.json"
DATASET_TRAIN_MINI_EXTRACTIONS_FEW_SHOT_PATH="../../../data/processed/stanford/ace_train_triplets_filtered_merged.json"
DATASET_TEST_MINI_EXTRACTIONS_FEW_SHOT_PATH="../../../data/processed/stanford/ace_test_triplets_filtered_merged.json"
EPOCHS=10
MULTIPLICATIVE_LR=0.99
BATCH_SIZE=32
BATCH_SIZE_FEW_SHOT=5
AVERAGE_OVER_RUNS_NUM=5
FEW_SHOT_EXAMPLES_NUM=500
TRAINING_TASK="trigger identification"
SEED=$1
NAME="STANFORD_roberta_ti_pretrained_maven_sequential_train_on_ace_500_shot_implicit_multitask_predict_on_ace_average_over_5_runs_SEED=${SEED}"
DEVICE="cuda:0"
LOAD_PRETRAINED_MODEL_FROM="../../../models_seed_${SEED}/maven_implicit_multitask_model_grid_search_STANFORD"
LABEL_HIDDEN_SIZE=10

python ../../../baselines/ti_rl_sequential_implicit_multitask_training_few_shot_target_evaluation.py \
	--lr $LR \
	--model "$MODEL" \
	--do_lower_case $DO_LOWER_CASE \
	--dataset_the_model_was_pretrained_on "$DATASET_THE_MODEL_WAS_PRETRAINED_ON" \
	--dataset_to_train_and_few_shot_evaluate_on "$DATASET_TO_TRAIN_AND_FEW_SHOT_EVALUATE_ON" \
	--dataset_train_few_shot_path "$DATASET_TRAIN_FEW_SHOT_PATH" \
	--dataset_test_few_shot_path "$DATASET_TEST_FEW_SHOT_PATH" \
	--dataset_train_mini_extractions_few_shot_path "$DATASET_TRAIN_MINI_EXTRACTIONS_FEW_SHOT_PATH" \
	--dataset_test_mini_extractions_few_shot_path "$DATASET_TEST_MINI_EXTRACTIONS_FEW_SHOT_PATH" \
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
    --load_pretrained_model_from "$LOAD_PRETRAINED_MODEL_FROM" \
	--label_hidden_size $LABEL_HIDDEN_SIZE