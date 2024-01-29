#!/bin/bash

LR=0.00001
MODEL="roberta-base"
DO_LOWER_CASE=False
DATASET_TO_TRAIN_ON="MAVEN"
DATASET_TO_ZERO_SHOT_EVALUATE_ON="EVEXTRA"
DATASET_TRAIN_PATH="../../../data/raw/maven/train.jsonl"
DATASET_VALID_PATH="N/A"
DATASET_TEST_PATH="../../../data/raw/maven/valid.jsonl"
DATASET_TRAIN_MINI_EXTRACTIONS_PATH="../../../data/processed/mini/maven_train_triplets_filtered_merged.json"
DATASET_VALID_MINI_EXTRACTIONS_PATH="N/A"
DATASET_TEST_MINI_EXTRACTIONS_PATH="../../../data/processed/mini/maven_test_triplets_filtered_merged.json"
DATASET_VALID_TO_EVALUATE_ON_PATH="../../../data/processed/evextra/valid.json"
DATASET_TEST_TO_EVALUATE_ON_PATH="../../../data/processed/evextra/test.json"
EPOCHS=10
MULTIPLICATIVE_LR=0.99
BATCH_SIZE=28
TRAINING_TASK="trigger identification"
SEED=$1
NAME="roberta_ti_rl_mlm_train_on_maven_multitask_two_heads_zero_shot_mlm_validation_predict_on_evextra_SEED=${SEED}"
DEVICE="cuda:0"
SAVE_TO="../../../models_seed_${SEED}/maven_multitask_evextra_mlm_validation_model_two_heads"
MLM_CHUNK_SIZE=128
MLM_PROBA=0.15

python ../../../baselines/mlm_validation/ti_rl_mlm_multitask_two_heads_training_zero_shot_target_evaluation.py \
	--lr $LR \
	--model "$MODEL" \
	--do_lower_case $DO_LOWER_CASE \
	--dataset_to_train_on "$DATASET_TO_TRAIN_ON" \
	--dataset_to_zero_shot_evaluate_on "$DATASET_TO_ZERO_SHOT_EVALUATE_ON" \
	--dataset_train_path "$DATASET_TRAIN_PATH" \
	--dataset_valid_path "$DATASET_VALID_PATH" \
	--dataset_test_path "$DATASET_TEST_PATH" \
	--dataset_valid_to_evaluate_on_path "$DATASET_VALID_TO_EVALUATE_ON_PATH" \
	--dataset_test_to_evaluate_on_path "$DATASET_TEST_TO_EVALUATE_ON_PATH" \
	--dataset_train_mini_extractions_path "$DATASET_TRAIN_MINI_EXTRACTIONS_PATH" \
	--dataset_valid_mini_extractions_path "$DATASET_VALID_MINI_EXTRACTIONS_PATH" \
	--dataset_test_mini_extractions_path "$DATASET_TEST_MINI_EXTRACTIONS_PATH" \
	--epochs $EPOCHS \
	--multiplicative_lr $MULTIPLICATIVE_LR \
	--batch_size $BATCH_SIZE \
	--training_task "$TRAINING_TASK" \
	--name "$NAME" \
	--seed $SEED \
	--device "$DEVICE" \
	--save_to "$SAVE_TO" \
	--mlm_chunk_size $MLM_CHUNK_SIZE \
	--mlm_proba $MLM_PROBA