# author: ddukic

import math
from statistics import mean

import evaluate
import torch
from constants import OTHER

seqeval = evaluate.load("seqeval")


def build_vocab(labels, BIO_tagging=True):
    # OTHER token has idx 0
    all_labels = [OTHER]
    for label in labels:
        if BIO_tagging:
            all_labels.append("B-{}".format(label))
            all_labels.append("I-{}".format(label))
        else:
            all_labels.append(label)
    label2id = {tag: id for id, tag in enumerate(all_labels)}
    id2label = {id: tag for id, tag in enumerate(all_labels)}

    return all_labels, id2label, label2id


def get_perplexity(losses):
    try:
        perplexity = math.exp(mean(losses))
    except OverflowError:
        perplexity = float("inf")

    return perplexity


def compute_metrics(model, loader):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            pred = torch.argmax(out.logits, dim=-1)
            target = batch["labels"]
            predictions.extend(pred.tolist())
            targets.extend(target.tolist())

    predictions = [
        [model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, targets)
    ]

    targets = [
        [model.config.id2label[l] for l in label if l != -100] for label in targets
    ]

    all_metrics_classification = seqeval.compute(
        predictions=predictions, references=targets, scheme="IOB2", mode="strict"
    )
    return {"identification": all_metrics_classification}


def compute_metrics_multitask(model, loader, labels_key):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels_key=labels_key,
            )
            pred = torch.argmax(out.logits, dim=-1)
            target = batch[labels_key]
            predictions.extend(pred.tolist())
            targets.extend(target.tolist())

    predictions = [
        [
            model.id2labels[labels_key][p]
            for (p, l) in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(predictions, targets)
    ]

    targets = [
        [model.id2labels[labels_key][l] for l in label if l != -100]
        for label in targets
    ]

    all_metrics_classification = seqeval.compute(
        predictions=predictions, references=targets, scheme="IOB2", mode="strict"
    )
    return {"identification": all_metrics_classification}


def logging(model, loader, loss, losses, epoch, mode="train", normalized=False):
    if normalized:
        loss_string = mode + " loss: " + str(round(loss, 3))
    else:
        loss_string = mode + " loss: " + str(round(loss / len(loader.dataset), 3))

    all_metrics = compute_metrics(model, loader)

    identification_string = "identification" + "\nP={:.3f}\tR={:.3f}\tF1={:.3f}".format(
        all_metrics["identification"]["overall_precision"],
        all_metrics["identification"]["overall_recall"],
        all_metrics["identification"]["overall_f1"],
    )

    f1_string = (
        mode + " F1: " + str(round(all_metrics["identification"]["overall_f1"], 3))
    )

    perp = get_perplexity(losses)

    perp_string = mode + " perplexity: " + str(round(perp, 3))

    epoch_string = "\033[1m" + "Epoch: " + str(epoch) + "\033[0m"

    if mode == "train":
        print(epoch_string)
    print(loss_string)
    print(f1_string)
    print("------------------------")
    print(mode + " evaluation details")
    print("------------------------")
    print(identification_string)
    print("------------------------")
    print(perp_string)
    print("------------------------")

    return all_metrics, perp


def logging_multitask(
    model, loader, loss, losses, epoch, mode="train", normalized=False
):
    if normalized:
        if mode == "train":
            loss_string = mode + " averaged loss: " + str(round(loss, 3))
        else:
            loss_string = mode + " trigger loss: " + str(round(loss, 3))
    else:
        if mode == "train":
            loss_string = (
                mode + " averaged loss: " + str(round(loss / len(loader.dataset), 3))
            )
        else:
            loss_string = (
                mode + " trigger loss: " + str(round(loss / len(loader.dataset), 3))
            )

    all_metrics_trigger = compute_metrics_multitask(model, loader, "labels")
    all_metrics_relation = compute_metrics_multitask(model, loader, "label_type_ids")

    identification_string_trigger = (
        "identification trigger"
        + "\nP={:.3f}\tR={:.3f}\tF1={:.3f}".format(
            all_metrics_trigger["identification"]["overall_precision"],
            all_metrics_trigger["identification"]["overall_recall"],
            all_metrics_trigger["identification"]["overall_f1"],
        )
    )

    identification_string_relation = (
        "identification relation"
        + "\nP={:.3f}\tR={:.3f}\tF1={:.3f}".format(
            all_metrics_relation["identification"]["overall_precision"],
            all_metrics_relation["identification"]["overall_recall"],
            all_metrics_relation["identification"]["overall_f1"],
        )
    )

    f1_string_trigger = (
        mode
        + " trigger F1: "
        + str(round(all_metrics_trigger["identification"]["overall_f1"], 3))
    )

    f1_string_relation = (
        mode
        + " relation F1: "
        + str(round(all_metrics_relation["identification"]["overall_f1"], 3))
    )

    perp = get_perplexity(losses)

    perp_string = mode + " perplexity: " + str(round(perp, 3))

    epoch_string = "\033[1m" + "Epoch: " + str(epoch) + "\033[0m"

    if mode == "train":
        print(epoch_string)
    print(loss_string)
    print(f1_string_trigger)
    print(f1_string_relation)
    print("------------------------")
    print(mode + " evaluation details")
    print("------------------------")
    print(identification_string_trigger)
    print("------------------------")
    print(identification_string_relation)
    print("------------------------")
    print(perp_string)
    print("------------------------")

    return all_metrics_trigger, all_metrics_relation, perp


class CustomCollatorMLM(object):
    # ensure reproducibility
    def __init__(self, collator, seed):
        self.seed = seed
        self.collator = collator

    def __call__(self, batch):
        torch.manual_seed(self.seed)
        collated = self.collator(batch)

        return collated
