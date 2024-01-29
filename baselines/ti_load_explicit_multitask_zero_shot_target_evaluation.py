# author: ddukic

import argparse
import os
import random

import numpy as np
import torch
import wandb
from dataset import ACE2005Dataset, EDNYTDataset, EVEXTRADataset, MavenDataset
from torch.utils import data
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from util import build_vocab, compute_metrics_multitask

from models import (
    RobertaForMultiTaskOneHeadTokenClassification,
    RobertaForMultiTaskTwoHeadsTokenClassification,
)

# Ensure reproducibility on CUDA
# set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance)
# or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

dataset_mapping = {
    "ACE2005": ACE2005Dataset,
    "MAVEN": MavenDataset,
    "EDNYT": EDNYTDataset,
    "EVEXTRA": EVEXTRADataset,
}


def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False


parser = argparse.ArgumentParser(
    description="Runs trigger sequence labeling inference on target with pretrained model on source"
)
parser.add_argument(
    "--model",
    type=str,
    help="Hugging face model identifier (for now 'bert-base-cased' or 'roberta-base')",
)
parser.add_argument(
    "--do_lower_case",
    type=parse_boolean,
    help="Transformer tokenizer option for lowercasing",
)
parser.add_argument(
    "--dataset_the_model_was_pretrained_on",
    type=str,
    help="The dataset which was used to train the model on source",
)
parser.add_argument(
    "--dataset_to_zero_shot_evaluate_on",
    type=str,
    help="The dataset to zero shot evaluate on name",
)
parser.add_argument(
    "--dataset_target_to_evaluate_on_path",
    type=str,
    help="The test dataset for target domain inference",
)
parser.add_argument("--batch_size", type=int, help="Size of a batch")
parser.add_argument(
    "--training_task",
    type=str,
    help="Training task specification",
)
parser.add_argument(
    "--seed",
    type=int,
    help="Seed to ensure reproducibility by using the same seed everywhere",
)
parser.add_argument(
    "--name",
    type=str,
    help="Name of experiment on W&B",
)
parser.add_argument("--device", type=str, help="GPU device to run training on")
parser.add_argument(
    "--load_pretrained_model_from",
    type=str,
    help="Path where to load pretrained model from",
)
parser.add_argument(
    "--token_classification_heads_num", type=str, help="Can be 'one' or 'two'"
)
args = parser.parse_args()

# gotta count add_argument calls
all_passed = sum([v is not None for k, v in vars(args).items()]) == len(vars(args))

print("All arguments passed?", all_passed)

if not all_passed:
    exit(1)

wandb.init(
    project="oee-paper",
    entity="ddukic",
    name=args.name,
    config=args,
)

config = wandb.config

wandb.define_metric("epoch")
wandb.define_metric("train/epoch*", step_metric="epoch")
wandb.define_metric("valid/epoch*", step_metric="epoch")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def evaluate(
    model,
    loader_zero_shot,
):
    all_metrics_zero_shot = compute_metrics_multitask(model, loader_zero_shot, "labels")

    wandb.summary["test_all_metrics_zero_shot"] = all_metrics_zero_shot

    print("test_all_metrics_zero_shot", all_metrics_zero_shot)


if __name__ == "__main__":
    set_seed(config.seed)

    if "roberta" in config.model:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            do_lower_case=config.do_lower_case,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            do_lower_case=config.do_lower_case,
        )

    all_labels_trigger, id2trigger, trigger2id = build_vocab(["Trigger"])
    _, id2relation, relation2id = build_vocab(["Relation"])

    if config.token_classification_heads_num == "one":
        # check if the model is loaded from source pretrained or vanilla transformer
        if config.model != config.load_pretrained_model_from:
            bert_model = RobertaForMultiTaskOneHeadTokenClassification.from_pretrained(
                config.load_pretrained_model_from,
                id2labels={"labels": id2trigger, "label_type_ids": id2relation},
            )
        else:
            bert_model = RobertaForMultiTaskOneHeadTokenClassification.from_pretrained(
                config.load_pretrained_model_from,
                num_labels=len(all_labels_trigger),
                id2label=id2trigger,
                label2id=trigger2id,
                id2labels={"labels": id2trigger, "label_type_ids": id2relation},
            )
    else:
        # check if the model is loaded from source pretrained or vanilla transformer
        if config.model != config.load_pretrained_model_from:
            bert_model = RobertaForMultiTaskTwoHeadsTokenClassification.from_pretrained(
                config.load_pretrained_model_from,
                id2labels={"labels": id2trigger, "label_type_ids": id2relation},
            )
        else:
            bert_model = RobertaForMultiTaskTwoHeadsTokenClassification.from_pretrained(
                config.load_pretrained_model_from,
                num_labels=len(all_labels_trigger),
                id2label=id2trigger,
                label2id=trigger2id,
                id2labels={"labels": id2trigger, "label_type_ids": id2relation},
            )
    # pad to max sequence in batch
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True
    )

    target_dataset = dataset_mapping[config.dataset_to_zero_shot_evaluate_on]

    zero_shot_test = target_dataset(
        fpath=config.dataset_target_to_evaluate_on_path,
        tokenizer=tokenizer,
        trigger2id=trigger2id,
        task=config.training_task,
    )

    loader_zero_shot_test = data.DataLoader(
        dataset=zero_shot_test,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed),
    )

    device = config.device if torch.cuda.is_available() else "cpu"

    print(device)

    # prepare model for training
    bert_model = bert_model.to(device)

    evaluate(
        model=bert_model,
        loader_zero_shot=loader_zero_shot_test,
    )

    wandb.finish()
