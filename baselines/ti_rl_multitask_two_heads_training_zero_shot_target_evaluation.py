# author: ddukic

import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import wandb
from dataset import (
    ACE2005Dataset,
    ACE2005TriggerRelationDataset,
    EDNYTDataset,
    EDNYTTriggerRelationDataset,
    EVEXTRADataset,
    EVEXTRATriggerRelationDataset,
    MavenDataset,
    MavenTriggerRelationDataset,
)
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from util import build_vocab, compute_metrics_multitask, logging_multitask

from models import RobertaForMultiTaskTwoHeadsTokenClassification

# Ensure reproducibility on CUDA
# set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance)
# or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

dataset_mapping_trigger = {
    "ACE2005": ACE2005Dataset,
    "MAVEN": MavenDataset,
    "EDNYT": EDNYTDataset,
    "EVEXTRA": EVEXTRADataset,
}
dataset_mapping_trigger_relation = {
    "ACE2005": ACE2005TriggerRelationDataset,
    "MAVEN": MavenTriggerRelationDataset,
    "EDNYT": EDNYTTriggerRelationDataset,
    "EVEXTRA": EVEXTRATriggerRelationDataset,
}


def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False


parser = argparse.ArgumentParser(
    description="Runs trigger and relation sequence labeling training on source dataset with inference on target"
)
parser.add_argument("--lr", type=float, help="Learning rate for optimizer")
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
    "--dataset_to_train_on",
    type=str,
    help="The dataset to train on name",
)
parser.add_argument(
    "--dataset_to_zero_shot_evaluate_on",
    type=str,
    help="The dataset to zero shot evaluate on name",
)
parser.add_argument(
    "--dataset_train_path",
    type=str,
    help="The train dataset for source domain training",
)
parser.add_argument(
    "--dataset_valid_path",
    type=str,
    help="The valid dataset for source domain training",
)
parser.add_argument(
    "--dataset_test_path",
    type=str,
    help="The test dataset for source domain training",
)
parser.add_argument(
    "--dataset_train_mini_extractions_path",
    type=str,
    help="The train relation dataset for source domain training",
)
parser.add_argument(
    "--dataset_valid_mini_extractions_path",
    type=str,
    help="The valid relation dataset for source domain training",
)
parser.add_argument(
    "--dataset_test_mini_extractions_path",
    type=str,
    help="The test relation dataset for source domain training",
)
parser.add_argument(
    "--dataset_test_to_evaluate_on_path",
    type=str,
    help="The test dataset for target domain inference",
)
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument(
    "--multiplicative_lr",
    type=float,
    help="A factor for multiplying learning rate in each epoch",
)
parser.add_argument("--batch_size", type=int, help="Size of a batch")
parser.add_argument(
    "--training_task",
    type=str,
    help="Training task that is going to be used for zero shot evaluation",
)
parser.add_argument(
    "--name",
    type=str,
    help="Name of experiment on W&B",
)
parser.add_argument(
    "--seed",
    type=int,
    help="Seed to ensure reproducibility by using the same seed everywhere",
)
parser.add_argument(
    "--save_to",
    type=str,
    help="Path where to save the trained model",
)
parser.add_argument("--device", type=str, help="GPU device to run training on")
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


def train(
    model,
    loader_train,
    loader_valid,
    optimizer,
    criterion,
    scheduler,
    epochs,
):
    best_valid_score = 0.0
    best_model = copy.deepcopy(model.state_dict())

    for i in tqdm(range(epochs)):
        train_loss = calculate_epoch_loss(
            model=model,
            loader=loader_train,
            optimizer=optimizer,
            criterion=criterion,
            mode="train",
        )

        scheduler.step()

        valid_loss = calculate_epoch_loss(
            model=model,
            loader=loader_valid,
            optimizer=optimizer,
            criterion=criterion,
            mode="valid",
        )

        train_all_metrics_trigger, train_all_metrics_relation = logging_multitask(
            model=model,
            loader=loader_train,
            loss=train_loss,
            epoch=i,
            mode="train",
            normalized=True,
        )

        valid_all_metrics_trigger, valid_all_metrics_relation = logging_multitask(
            model=model,
            loader=loader_valid,
            loss=valid_loss,
            epoch=i,
            mode="valid",
            normalized=False,
        )

        # x axis goes from 0 to 9 because of wandb
        train_metrics = {
            "train/epoch_loss": train_loss,
            "train/epoch_f1_score_trigger": train_all_metrics_trigger["identification"][
                "overall_f1"
            ],
            "train/epoch_f1_score_relation": train_all_metrics_relation[
                "identification"
            ]["overall_f1"],
            "epoch": i,
        }

        valid_metrics = {
            "valid/epoch_loss": valid_loss / len(loader_valid.dataset),
            "valid/epoch_f1_score_trigger": valid_all_metrics_trigger["identification"][
                "overall_f1"
            ],
            "valid/epoch_f1_score_relation": valid_all_metrics_relation[
                "identification"
            ]["overall_f1"],
            "epoch": i,
        }

        wandb.log({**train_metrics, **valid_metrics})

        valid_f1 = valid_all_metrics_trigger["identification"]["overall_f1"]

        if valid_f1 > best_valid_score:
            best_model = copy.deepcopy(model.state_dict())
            best_valid_score = valid_f1

    return best_model, best_valid_score


def collect_losses(model, criterion, batch, labels_key="labels"):
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels_key=labels_key,
    )

    # model.num_labels is equal to out.logits.shape[-1]
    tag_loss = criterion(
        out.logits.view(-1, model.num_labels), batch[labels_key].view(-1)
    )

    return tag_loss


def calculate_epoch_loss(
    model,
    loader,
    optimizer,
    criterion,
    mode="train",
):
    loss = 0.0
    if mode == "train":
        model.train()

        loss_1 = 0.0
        loss_2 = 0.0

        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model.zero_grad()
            optimizer.zero_grad()

            trigger_loss = collect_losses(model, criterion, batch, "labels")
            relation_loss = collect_losses(model, criterion, batch, "label_type_ids")

            loss_1 += trigger_loss.item()
            loss_2 += relation_loss.item()

            avg_loss = (
                trigger_loss / batch["labels"].shape[0]
                + relation_loss / batch["labels"].shape[0]
            ) / 2

            avg_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        loss = (loss_1 / len(loader.dataset) + loss_2 / len(loader.dataset)) / 2
    else:
        model.eval()

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                trigger_loss = collect_losses(model, criterion, batch, "labels")

                loss += trigger_loss.item()

    return loss


def train_evaluate(
    model,
    loader_train,
    loader_valid,
    loader_test,
    loader_zero_shot,
    optimizer,
    criterion,
    scheduler,
    epochs,
    save_to,
):
    best_model, best_valid_score = train(
        model=model,
        loader_train=loader_train,
        loader_valid=loader_valid,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=epochs,
    )

    model.load_state_dict(best_model)

    model.save_pretrained(save_to)

    all_metrics_trigger = compute_metrics_multitask(model, loader_test, "labels")
    all_metrics_relation = compute_metrics_multitask(
        model, loader_test, "label_type_ids"
    )
    all_metrics_zero_shot_trigger = compute_metrics_multitask(
        model, loader_zero_shot, "labels"
    )

    wandb.summary["valid_f1_trigger"] = best_valid_score
    wandb.summary["test_all_metrics_trigger"] = all_metrics_trigger
    wandb.summary["test_all_metrics_relation"] = all_metrics_relation
    wandb.summary["test_all_metrics_zero_shot"] = all_metrics_zero_shot_trigger

    print("test_all_metrics_trigger", all_metrics_trigger)
    print("test_all_metrics_relation", all_metrics_relation)
    print("test_all_metrics_zero_shot", all_metrics_zero_shot_trigger)


if __name__ == "__main__":
    set_seed(config.seed)

    if "roberta" in config.model:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model, do_lower_case=config.do_lower_case, add_prefix_space=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            do_lower_case=config.do_lower_case,
        )

    all_labels_trigger, id2trigger, trigger2id = build_vocab(["Trigger"])
    all_labels_relation, id2relation, relation2id = build_vocab(["Relation"])

    # label_type_ids are relation ids
    # all_labels_trigger and all_labels_relation have same length
    bert_model = RobertaForMultiTaskTwoHeadsTokenClassification.from_pretrained(
        config.model,
        num_labels=len(all_labels_trigger),
        id2label=id2trigger,
        label2id=trigger2id,
        id2labels={"labels": id2trigger, "label_type_ids": id2relation},
    )
    # pad to max sequence in batch
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True
    )

    source_dataset = dataset_mapping_trigger_relation[config.dataset_to_train_on]
    target_dataset = dataset_mapping_trigger[config.dataset_to_zero_shot_evaluate_on]

    if config.dataset_to_train_on == "MAVEN":
        dataset_train = source_dataset(
            fpath_trigger=config.dataset_train_path,
            fpath_relation=config.dataset_train_mini_extractions_path,
            tokenizer=tokenizer,
            trigger2id=trigger2id,
            relation2id=relation2id,
        )
        len_train = int(0.8 * len(dataset_train))

        # maven dataset does not have gold labels for test set, so we use valid set as test set, and sample a valid set from the train set
        dataset_train, dataset_valid = data.random_split(
            dataset_train,
            [int(0.8 * len(dataset_train)), len(dataset_train) - len_train],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        dataset_train = source_dataset(
            fpath_trigger=config.dataset_train_path,
            fpath_relation=config.dataset_train_mini_extractions_path,
            tokenizer=tokenizer,
            trigger2id=trigger2id,
            relation2id=relation2id,
        )
        dataset_valid = source_dataset(
            fpath_trigger=config.dataset_valid_path,
            fpath_relation=config.dataset_valid_mini_extractions_path,
            tokenizer=tokenizer,
            trigger2id=trigger2id,
            relation2id=relation2id,
        )

    dataset_test = source_dataset(
        fpath_trigger=config.dataset_test_path,
        fpath_relation=config.dataset_test_mini_extractions_path,
        tokenizer=tokenizer,
        trigger2id=trigger2id,
        relation2id=relation2id,
    )
    zero_shot_test = target_dataset(
        fpath=config.dataset_test_to_evaluate_on_path,
        tokenizer=tokenizer,
        trigger2id=trigger2id,
        task=config.training_task,
    )

    loader_train = data.DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed),
    )

    loader_valid = data.DataLoader(
        dataset=dataset_valid,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed),
    )

    loader_test = data.DataLoader(
        dataset=dataset_test,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed),
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

    optimizer = torch.optim.Adam(bert_model.parameters(), lr=config.lr)

    # notice reduction argument
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer=optimizer, lr_lambda=lambda x: config.multiplicative_lr
    )

    train_evaluate(
        model=bert_model,
        loader_train=loader_train,
        loader_valid=loader_valid,
        loader_test=loader_test,
        loader_zero_shot=loader_zero_shot_test,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=config.epochs,
        save_to=config.save_to,
    )

    wandb.finish()
