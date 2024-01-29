#!/bin/bash

MODEL="roberta-base"
DO_LOWER_CASE=False
DATASET_THE_MODEL_WAS_PRETRAINED_ON="MAVEN"
DATASET_TO_ZERO_SHOT_EVALUATE_ON="EVEXTRA"
DATASET_TARGET_TO_EVALUATE_ON_PATH="../../../data/processed/evextra/test.json"
BATCH_SIZE=32
TRAINING_TASK="trigger identification"
SEED=$1
NAME="STANFORD_roberta_ti_two_heads_multitask_pretrained_maven_zero_shot_predict_on_evextra_SEED=${SEED}"
DEVICE="cuda:0"
LOAD_PRETRAINED_MODEL_FROM="../../../models_seed_${SEED}/maven_multitask_model_two_heads_STANFORD"
TOKEN_CLASSIFICATION_HEADS_NUM="two"

python ../../../baselines/ti_load_explicit_multitask_zero_shot_target_evaluation.py \
	--model "$MODEL" \
	--do_lower_case $DO_LOWER_CASE \
	--dataset_the_model_was_pretrained_on "$DATASET_THE_MODEL_WAS_PRETRAINED_ON" \
	--dataset_to_zero_shot_evaluate_on "$DATASET_TO_ZERO_SHOT_EVALUATE_ON" \
	--dataset_target_to_evaluate_on_path "$DATASET_TARGET_TO_EVALUATE_ON_PATH" \
	--batch_size $BATCH_SIZE \
	--training_task "$TRAINING_TASK" \
	--seed $SEED \
	--name "$NAME" \
	--device "$DEVICE" \
	--load_pretrained_model_from "$LOAD_PRETRAINED_MODEL_FROM" \
	--token_classification_heads_num "$TOKEN_CLASSIFICATION_HEADS_NUM"