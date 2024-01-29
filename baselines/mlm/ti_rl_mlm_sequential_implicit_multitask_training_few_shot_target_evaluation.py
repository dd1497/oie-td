# author: ddukic

import argparse
import copy
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import wandb
from dataset import (
    ACE2005MLMDataset,
    ACE2005TriggerRelationDataset,
    EDNYTMLMDataset,
    EDNYTTriggerRelationDataset,
    EVEXTRAMLMDataset,
    EVEXTRATriggerRelationDataset,
    MavenMLMDataset,
    MavenTriggerRelationDataset,
)
from models import RobertaForMLMTriggerRelationTokenClassification
from torch.utils import data
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
)
from util import CustomCollatorMLM, build_vocab, compute_metrics, logging

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
    "ACE2005": ACE2005TriggerRelationDataset,
    "MAVEN": MavenTriggerRelationDataset,
    "EDNYT": EDNYTTriggerRelationDataset,
    "EVEXTRA": EVEXTRATriggerRelationDataset,
}

dataset_mlm_mapping = {
    "ACE2005": ACE2005MLMDataset,
    "MAVEN": MavenMLMDataset,
    "EDNYT": EDNYTMLMDataset,
    "EVEXTRA": EVEXTRAMLMDataset,
}


def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False


parser = argparse.ArgumentParser(
    description="Runs few-shot trigger and relation (with target instances) sequence labeling with MLM training on a pretrained (source) multitask model and evaluates on target"
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
    "--dataset_the_model_was_pretrained_on",
    type=str,
    help="The dataset which was used to train the initial model",
)
parser.add_argument(
    "--dataset_to_train_and_few_shot_evaluate_on",
    type=str,
    help="The dataset to draw few shot examples from and evaluate on name",
)
parser.add_argument(
    "--dataset_train_few_shot_path",
    type=str,
    help="The dataset to draw few shot examples from",
)
parser.add_argument(
    "--dataset_test_few_shot_path",
    type=str,
    help="The test dataset for target domain inference",
)
parser.add_argument(
    "--dataset_train_mini_extractions_few_shot_path",
    type=str,
    help="The train relation dataset for drawing few shot examples",
)
parser.add_argument(
    "--dataset_test_mini_extractions_few_shot_path",
    type=str,
    help="The test relation dataset for target domain inference",
)
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument(
    "--multiplicative_lr",
    type=float,
    help="A factor for multiplying learning rate in each epoch",
)
parser.add_argument(
    "--batch_size", type=int, help="Size of a batch for mlm and test evaluation"
)
parser.add_argument("--batch_size_few_shot", type=int, help="Size of a few shot batch")
parser.add_argument(
    "--average_over_runs_num",
    type=int,
    help="How many runs to do with different X-shot samples",
)
parser.add_argument(
    "--few_shot_examples_num",
    type=int,
    help="X-shot setting",
)
parser.add_argument(
    "--training_task",
    type=str,
    help="Training task specification",
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
    "--load_pretrained_model_from",
    type=str,
    help="Path where to load pretrained model from",
)
parser.add_argument("--device", type=str, help="GPU device to run training on")
parser.add_argument(
    "--label_hidden_size",
    type=int,
    help="Hidden size of relation label embedding matrix",
)
parser.add_argument("--mlm_chunk_size", type=int, help="length of MLM chunks")
parser.add_argument(
    "--mlm_proba", type=float, help="Probability of masking tokens for MLM"
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
    loader_few_shot_train,
    loader_mlm_train,
    optimizer,
    criterion,
    scheduler,
    epochs,
    configuration_string,
):
    for i in tqdm(range(epochs)):
        train_loss, train_losses_mlm = calculate_epoch_loss(
            model=model,
            loader=loader_few_shot_train,
            loader_mlm=loader_mlm_train,
            optimizer=optimizer,
            criterion=criterion,
            mode="train",
        )

        scheduler.step()

        train_all_metrics_trigger, train_perp = logging(
            model=model,
            loader=loader_few_shot_train,
            loss=train_loss,
            losses=train_losses_mlm,
            epoch=i,
            mode="train",
            normalized=False,
        )

        # x axis goes from 0 to 9 because of wandb
        train_metrics = {
            "train/epoch_loss "
            + configuration_string: train_loss / len(loader_few_shot_train.dataset),
            "train/epoch_f1_score_trigger "
            + configuration_string: train_all_metrics_trigger["identification"][
                "overall_f1"
            ],
            "train/epoch_perplexity_score": train_perp,
            "epoch": i,
        }

        wandb.log({**train_metrics})

    # return last model
    return copy.deepcopy(model.state_dict())


def collect_losses(model, criterion, batch):
    out = model(**batch)

    # model.num_labels is equal to out.logits.shape[-1]
    tag_loss = criterion(
        out.logits.view(-1, model.num_labels), batch["labels"].view(-1)
    )

    return tag_loss


def collect_losses_mlm(model, criterion, batch):
    out = model.forward_mlm(**batch)

    # model.num_labels is equal to out.logits.shape[-1]
    mlm_loss = criterion(
        out.logits.view(-1, model.config.vocab_size), batch["labels"].view(-1)
    )

    return mlm_loss


def calculate_epoch_loss(
    model,
    loader,
    loader_mlm,
    optimizer,
    criterion,
    mode="train",
):
    loss = 0.0
    losses_mlm = []
    if mode == "train":
        model.train()

        for batch in loader_mlm:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model.zero_grad()
            optimizer.zero_grad()

            mlm_loss = collect_losses_mlm(model, criterion, batch)

            losses_mlm.append(mlm_loss.item())

            mlm_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model.zero_grad()
            optimizer.zero_grad()

            tag_loss = collect_losses(model, criterion, batch)

            loss += tag_loss.item() * batch["labels"].shape[0]

            tag_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
    else:
        model.eval()

        with torch.no_grad():
            for batch in loader_mlm:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                mlm_loss = collect_losses_mlm(model, criterion, batch)

                losses_mlm.append(mlm_loss.item())

            for batch in loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                tag_loss = collect_losses(model, criterion, batch)

                loss += tag_loss.item() * batch["labels"].shape[0]

    return loss, losses_mlm


def train_evaluate(
    model,
    loader_few_shot_train,
    loader_few_shot_test,
    loader_mlm_train,
    optimizer,
    criterion,
    scheduler,
    epochs,
    configuration_string,
):
    best_model = train(
        model=model,
        loader_few_shot_train=loader_few_shot_train,
        loader_mlm_train=loader_mlm_train,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=epochs,
        configuration_string=configuration_string,
    )

    model.load_state_dict(best_model)

    all_metrics_few_shot_test = compute_metrics(model, loader_few_shot_test)

    wandb.summary[
        "test_all_metrics_few_shot_trigger " + configuration_string
    ] = all_metrics_few_shot_test

    print(
        "test_all_metrics_few_shot_trigger",
        configuration_string,
        all_metrics_few_shot_test,
    )

    return all_metrics_few_shot_test


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

    # pad to max sequence in batch
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True
    )

    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=config.mlm_proba
    )

    fixed_seed_mlm_collator = CustomCollatorMLM(mlm_collator, seed=config.seed)

    target_dataset = dataset_mapping[config.dataset_to_train_and_few_shot_evaluate_on]

    mlm_dataset = dataset_mlm_mapping[config.dataset_to_train_and_few_shot_evaluate_on]

    few_shot_train = target_dataset(
        fpath_trigger=config.dataset_train_few_shot_path,
        fpath_relation=config.dataset_train_mini_extractions_few_shot_path,
        tokenizer=tokenizer,
        trigger2id=trigger2id,
        relation2id=relation2id,
        implicit=True,
    )

    few_shot_test = target_dataset(
        fpath_trigger=config.dataset_test_few_shot_path,
        fpath_relation=config.dataset_test_mini_extractions_few_shot_path,
        tokenizer=tokenizer,
        trigger2id=trigger2id,
        relation2id=relation2id,
        implicit=True,
    )

    mlm_dataset_train = mlm_dataset(
        fpath=config.dataset_train_few_shot_path,
        tokenizer=tokenizer,
        chunk_size=config.mlm_chunk_size,
    )

    # ensure to get something that is not all "O" tags
    sample_indices = [
        i
        for i, x in enumerate(few_shot_train)
        if trigger2id["B-Trigger"] in x["labels"]
    ]
    x_shot_samples = []
    seed_change = config.seed
    for i in range(config.average_over_runs_num):
        random.seed(seed_change)
        x_shot_indices = random.sample(sample_indices, config.few_shot_examples_num)
        x_shot_samples.append(x_shot_indices)
        print("Indices of few shot train examples", str(x_shot_indices))
        seed_change = seed_change // 2

    set_seed(config.seed)

    few_shot_samples_train = []
    for sample in x_shot_samples:
        few_shot_samples_train.append(data.Subset(few_shot_train, sample))

    loader_few_shot_test = data.DataLoader(
        dataset=few_shot_test,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed),
    )

    loader_mlm_train = data.DataLoader(
        dataset=mlm_dataset_train,
        batch_size=config.batch_size,
        collate_fn=fixed_seed_mlm_collator,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed),
    )

    device = config.device if torch.cuda.is_available() else "cpu"

    print(device)

    test_metrics_few_shot_averaged = defaultdict(float)

    for i, few_shot_dataset in enumerate(few_shot_samples_train):
        set_seed(config.seed)

        bert_model = RobertaForMLMTriggerRelationTokenClassification.from_pretrained(
            config.load_pretrained_model_from,
            num_labels=len(all_labels_trigger),
            id2label=id2trigger,
            label2id=trigger2id,
            label_type_size=len(all_labels_relation),
            label_hidden_size=config.label_hidden_size,
        )

        # prepare model for training
        bert_model = bert_model.to(device)

        print(
            "Check that the model loaded properly:",
            compute_metrics(bert_model, loader_few_shot_test),
        )

        print("------------------------")

        optimizer = torch.optim.Adam(bert_model.parameters(), lr=config.lr)

        # notice reduction argument
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer=optimizer, lr_lambda=lambda x: config.multiplicative_lr
        )

        loader_few_shot_train = data.DataLoader(
            dataset=few_shot_dataset,
            batch_size=config.batch_size_few_shot,
            collate_fn=data_collator,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config.seed),
            shuffle=True,
        )

        all_metrics_few_shot_test = train_evaluate(
            model=bert_model,
            loader_few_shot_train=loader_few_shot_train,
            loader_few_shot_test=loader_few_shot_test,
            loader_mlm_train=loader_mlm_train,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epochs=config.epochs,
            configuration_string="RUN " + str(i),
        )

        for key_average in [
            x
            for x in all_metrics_few_shot_test["identification"].keys()
            if "overall" in x
        ]:
            test_metrics_few_shot_averaged[
                key_average + "_averaged"
            ] += all_metrics_few_shot_test["identification"][key_average]

    test_metrics_few_shot_averaged = {
        k: v / config.average_over_runs_num
        for k, v in test_metrics_few_shot_averaged.items()
    }

    wandb.summary[
        "test_all_metrics_few_shot_trigger_averaged"
    ] = test_metrics_few_shot_averaged

    print("test_all_metrics_few_shot_trigger_averaged", test_metrics_few_shot_averaged)

    wandb.finish()
