#!/bin/bash

LR=0.00001
LR_EVENT_EMBS=(0.0001 0.00005 0.00001)
MODEL="roberta-base"
DO_LOWER_CASE=False
DATASET_TO_TRAIN_ON="MAVEN"
DATASET_TO_ZERO_SHOT_EVALUATE_ON="ACE2005"
DATASET_TRAIN_PATH="../../../data/raw/maven/train.jsonl"
DATASET_VALID_PATH="N/A"
DATASET_TEST_PATH="../../../data/raw/maven/valid.jsonl"
DATASET_TRAIN_MINI_EXTRACTIONS_PATH="../../../data/processed/stanford/maven_train_triplets_filtered_merged.json"
DATASET_VALID_MINI_EXTRACTIONS_PATH="N/A"
DATASET_TEST_MINI_EXTRACTIONS_PATH="../../../data/processed/stanford/maven_test_triplets_filtered_merged.json"
DATASET_TEST_TO_EVALUATE_ON_PATH="../../../data/raw/ace/test.json"
DATASET_TEST_MINI_EXTRACTIONS_RELATION_TO_EVALUATE_ON_PATH="../../../data/processed/stanford/ace_test_triplets_filtered_merged.json"
EPOCHS=10
MULTIPLICATIVE_LR=0.99
BATCH_SIZE=32
TRAINING_TASK="trigger identification"
SEED=$1
NAME="STANFORD_roberta_ti_train_on_maven_implicit_multitask_zero_shot_predict_on_ace_grid_search_SEED=${SEED}"
LABEL_HIDDEN_SIZES=(10 50 100 300)
DEVICE="cuda:0"
SAVE_TO="../../../models_seed_${SEED}/maven_implicit_multitask_model_grid_search_STANFORD"

python ../../../baselines/ti_rl_implicit_multitask_grid_search_training_zero_shot_target_evaluation.py \
	--lr $LR \
	--lr_event_embs ${LR_EVENT_EMBS[@]} \
	--model "$MODEL" \
	--do_lower_case $DO_LOWER_CASE \
	--dataset_to_train_on "$DATASET_TO_TRAIN_ON" \
	--dataset_to_zero_shot_evaluate_on "$DATASET_TO_ZERO_SHOT_EVALUATE_ON" \
	--dataset_train_path "$DATASET_TRAIN_PATH" \
	--dataset_valid_path "$DATASET_VALID_PATH" \
	--dataset_test_path "$DATASET_TEST_PATH" \
	--dataset_test_to_evaluate_on_path "$DATASET_TEST_TO_EVALUATE_ON_PATH" \
	--dataset_train_mini_extractions_path "$DATASET_TRAIN_MINI_EXTRACTIONS_PATH" \
	--dataset_valid_mini_extractions_path "$DATASET_VALID_MINI_EXTRACTIONS_PATH" \
	--dataset_test_mini_extractions_path "$DATASET_TEST_MINI_EXTRACTIONS_PATH" \
	--dataset_test_mini_extractions_relation_to_evaluate_on_path "$DATASET_TEST_MINI_EXTRACTIONS_RELATION_TO_EVALUATE_ON_PATH" \
	--epochs $EPOCHS \
	--multiplicative_lr $MULTIPLICATIVE_LR \
	--batch_size $BATCH_SIZE \
	--training_task "$TRAINING_TASK" \
	--name "$NAME" \
	--seed $SEED \
	--label_hidden_sizes ${LABEL_HIDDEN_SIZES[@]} \
	--device "$DEVICE" \
	--save_to "$SAVE_TO"