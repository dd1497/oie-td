# author: ddukic

import json

from constants import OTHER
from torch.utils import data


class OIEDataset(data.Dataset):
    def __init__(self, fpath, tokenizer, label2id, filter_by="relation", merged=True):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.tokens, self.tags, self.tag_ids = [], [], []

        visited = []

        with open(fpath, "r") as f:
            data = json.load(f)
            for item in data:
                sentence_tokens = data[item]["tokens"]
                sentence_tags = data[item]["bio_tags"]
                if merged:
                    if filter_by == "relation":
                        sentence_tags = [
                            tag if tag in ["B-Relation", "I-Relation"] else OTHER
                            for tag in sentence_tags
                        ]
                    pairs = []
                    for token, tag in zip(sentence_tokens, sentence_tags):
                        pairs.append((token, tag))

                    if pairs not in visited:
                        self.tokens.append(sentence_tokens)
                        self.tags.append(sentence_tags)
                        self.tag_ids.append([self.label2id[x] for x in sentence_tags])
                        visited.append(pairs)
                else:
                    for tag in sentence_tags:
                        if filter_by == "relation":
                            tag = [
                                t if t in ["B-Relation", "I-Relation"] else OTHER
                                for t in sentence_tags
                            ]
                        pairs = []
                        for token, t in zip(sentence_tokens, tag):
                            pairs.append((token, t))

                        if pairs not in visited:
                            self.tokens.append(sentence_tokens)
                            self.tags.append(tag)
                            self.tag_ids.append([self.label2id[x] for x in tag])
                            visited.append(pairs)

        self.tokenized_inputs = self.tokenize_and_align_labels(
            self.tokens, self.tag_ids
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_inputs["input_ids"][idx],
            "attention_mask": self.tokenized_inputs["attention_mask"][idx],
            "labels": self.tokenized_inputs["labels"][idx],
        }

    def tokenize_and_align_labels(self, tokens, tags_id):
        """Taken from https://huggingface.co/docs/transformers/tasks/token_classification"""
        tokenized_input = self.tokenizer(
            tokens, truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(tags_id):
            word_ids = tokenized_input.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_input["labels"] = labels
        return tokenized_input


class ACE2005Dataset(data.Dataset):
    def __init__(self, fpath, tokenizer, trigger2id, task="trigger classification"):
        self.tokenizer = tokenizer
        self.trigger2id = trigger2id
        self.tokens, self.triggers, self.trigger_ids = [], [], []

        with open(fpath, "r") as f:
            data = json.load(f)
            for item in data:
                words = item["words"]
                tags = [OTHER] * len(words)

                for event_mention in item["golden-event-mentions"]:
                    for i in range(
                        event_mention["trigger"]["start"],
                        event_mention["trigger"]["end"],
                    ):
                        if task == "trigger identification":
                            trigger_type = "Trigger"
                        else:
                            trigger_type = event_mention["event_type"]
                        if i == event_mention["trigger"]["start"]:
                            tags[i] = "B-{}".format(trigger_type)
                        else:
                            tags[i] = "I-{}".format(trigger_type)

                self.tokens.append(words)
                self.triggers.append(tags)
                self.trigger_ids.append([self.trigger2id[x] for x in tags])

        self.tokenized_inputs = self.tokenize_and_align_labels(
            self.tokens, self.trigger_ids
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_inputs["input_ids"][idx],
            "attention_mask": self.tokenized_inputs["attention_mask"][idx],
            "labels": self.tokenized_inputs["labels"][idx],
        }

    def tokenize_and_align_labels(self, tokens, tags_id):
        """Taken from https://huggingface.co/docs/transformers/tasks/token_classification"""
        tokenized_input = self.tokenizer(
            tokens, truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(tags_id):
            word_ids = tokenized_input.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_input["labels"] = labels
        return tokenized_input


class ACE2005TriggerRelationDataset(data.Dataset):
    def __init__(
        self,
        fpath_trigger,
        fpath_relation,
        tokenizer,
        trigger2id,
        relation2id,
        implicit=False,
    ):
        self.tokenizer = tokenizer
        self.trigger2id = trigger2id
        self.relation2id = relation2id
        (
            self.tokens,
            self.triggers,
            self.relations,
            self.trigger_ids,
            self.relation_ids,
        ) = ([], [], [], [], [])

        # is the multitask learning implicit or explicit
        self.implicit = implicit

        visited = []

        with open(fpath_trigger, "r") as f:
            trigger_data = json.load(f)

        with open(fpath_relation, "r") as f:
            relation_data = json.load(f)

        for i, item in enumerate(trigger_data):
            ace_tokens = item["words"]
            trigger_tags = [OTHER] * len(ace_tokens)
            for event_mention in item["golden-event-mentions"]:
                for j in range(
                    event_mention["trigger"]["start"],
                    event_mention["trigger"]["end"],
                ):
                    if j == event_mention["trigger"]["start"]:
                        trigger_tags[j] = "B-{}".format("Trigger")
                    else:
                        trigger_tags[j] = "I-{}".format("Trigger")

            if str(i) in relation_data:
                relation_tags = relation_data[str(i)]["bio_tags"]
                # can be multiple bio_tags for one sentence or one
                for relation_tag in relation_tags:
                    # remove Subject and Object info
                    relation_tag = [
                        tag if tag in ["B-Relation", "I-Relation"] else OTHER
                        for tag in relation_tag
                    ]
                    # handle duplicates
                    triple = []
                    for token, t_tag, r_tag in zip(
                        ace_tokens, trigger_tags, relation_tag
                    ):
                        triple.append((token, t_tag, r_tag))
                    if triple not in visited:
                        self.tokens.append(ace_tokens)
                        self.triggers.append(trigger_tags)
                        self.relations.append(relation_tag)
                        self.trigger_ids.append(
                            [self.trigger2id[x] for x in trigger_tags]
                        )
                        self.relation_ids.append(
                            [self.relation2id[x] for x in relation_tag]
                        )
            else:
                relation_tag = [OTHER] * len(ace_tokens)
                self.tokens.append(ace_tokens)
                self.triggers.append(trigger_tags)
                self.relations.append(relation_tag)
                self.trigger_ids.append([self.trigger2id[x] for x in trigger_tags])
                self.relation_ids.append([self.relation2id[x] for x in relation_tag])

        self.tokenized_inputs = self.tokenize_and_align_labels(
            self.tokens, self.trigger_ids, self.relation_ids
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_inputs["input_ids"][idx],
            "attention_mask": self.tokenized_inputs["attention_mask"][idx],
            "labels": self.tokenized_inputs["labels"][idx],
            "label_type_ids": self.tokenized_inputs["label_type_ids"][idx],
        }

    def tokenize_and_align_labels(self, tokens, tags_id, labels_type_id):
        tokenized_input = self.tokenizer(
            tokens,
            truncation=True,
            padding=True,
            is_split_into_words=True,
        )

        tokenized_input["labels"] = self.align_labels(
            tags_id, tokenized_input, ignore_index=-100
        )
        # set 0 instead of -100 for implicit case
        tokenized_input["label_type_ids"] = self.align_labels(
            labels_type_id, tokenized_input, ignore_index=0 if self.implicit else -100
        )
        return tokenized_input

    def align_labels(self, tags_id, tokenized_input, ignore_index):
        """Taken from https://huggingface.co/docs/transformers/tasks/token_classification"""
        labels = []
        for i, label in enumerate(tags_id):
            word_ids = tokenized_input.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(ignore_index)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(ignore_index)
                previous_word_idx = word_idx
            labels.append(label_ids)
        return labels


class MavenDataset(data.Dataset):
    def __init__(self, fpath, tokenizer, trigger2id, task="trigger classification"):
        self.tokenizer = tokenizer
        self.trigger2id = trigger2id
        self.tokens, self.triggers, self.trigger_ids = [], [], []

        with open(fpath, "r", encoding="utf-8") as f:
            data = [json.loads(ln) for ln in f.readlines()]
            for item in data:
                toks = [cnt["tokens"] for cnt in item["content"]]

                for sent_id, token_sent in enumerate(toks):
                    tags = [OTHER] * len(token_sent)

                    for event_item in item["events"]:
                        if task == "trigger identification":
                            trigger_type = "Trigger"
                        else:
                            trigger_type = event_item["type"]
                        for event_mention in event_item["mention"]:
                            if event_mention["sent_id"] == sent_id:
                                for i in range(
                                    event_mention["offset"][0],
                                    event_mention["offset"][1],
                                ):
                                    if i == event_mention["offset"][0]:
                                        tags[i] = "B-{}".format(trigger_type)
                                    else:
                                        tags[i] = "I-{}".format(trigger_type)

                    self.tokens.append(token_sent)
                    self.triggers.append(tags)
                    self.trigger_ids.append([self.trigger2id[x] for x in tags])

        self.tokenized_inputs = self.tokenize_and_align_labels(
            self.tokens, self.trigger_ids
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_inputs["input_ids"][idx],
            "attention_mask": self.tokenized_inputs["attention_mask"][idx],
            "labels": self.tokenized_inputs["labels"][idx],
        }

    def tokenize_and_align_labels(self, tokens, tags_id):
        """Taken from https://huggingface.co/docs/transformers/tasks/token_classification"""
        tokenized_input = self.tokenizer(
            tokens, truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(tags_id):
            word_ids = tokenized_input.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_input["labels"] = labels
        return tokenized_input


class MavenTriggerRelationDataset(data.Dataset):
    def __init__(
        self,
        fpath_trigger,
        fpath_relation,
        tokenizer,
        trigger2id,
        relation2id,
        implicit=False,
    ):
        self.tokenizer = tokenizer
        self.trigger2id = trigger2id
        self.relation2id = relation2id
        (
            self.tokens,
            self.triggers,
            self.relations,
            self.trigger_ids,
            self.relation_ids,
        ) = ([], [], [], [], [])

        # is the multitask learning implicit or explicit
        self.implicit = implicit

        visited = []

        with open(fpath_trigger, "r", encoding="utf-8") as f:
            trigger_data = [json.loads(ln) for ln in f.readlines()]

        with open(fpath_relation, "r") as f:
            relation_data = json.load(f)

        j = 0
        for item in trigger_data:
            tokens = [cnt["tokens"] for cnt in item["content"]]
            for sent_id, maven_tokens in enumerate(tokens):
                trigger_tags = [OTHER] * len(maven_tokens)
                for event_item in item["events"]:
                    for event_mention in event_item["mention"]:
                        if event_mention["sent_id"] == sent_id:
                            for i in range(
                                event_mention["offset"][0],
                                event_mention["offset"][1],
                            ):
                                if i == event_mention["offset"][0]:
                                    trigger_tags[i] = "B-{}".format("Trigger")
                                else:
                                    trigger_tags[i] = "I-{}".format("Trigger")
                if str(j) in relation_data:
                    relation_tags = relation_data[str(j)]["bio_tags"]
                    for relation_tag in relation_tags:
                        # remove Subject and Object info
                        relation_tag = [
                            tag if tag in ["B-Relation", "I-Relation"] else OTHER
                            for tag in relation_tag
                        ]
                        # handle duplicates
                        triple = []
                        for token, t_tag, r_tag in zip(
                            maven_tokens, trigger_tags, relation_tag
                        ):
                            triple.append((token, t_tag, r_tag))
                        if triple not in visited:
                            self.tokens.append(maven_tokens)
                            self.triggers.append(trigger_tags)
                            self.relations.append(relation_tag)
                            self.trigger_ids.append(
                                [self.trigger2id[x] for x in trigger_tags]
                            )
                            self.relation_ids.append(
                                [self.relation2id[x] for x in relation_tag]
                            )
                else:
                    relation_tag = [OTHER] * len(maven_tokens)
                    self.tokens.append(maven_tokens)
                    self.triggers.append(trigger_tags)
                    self.relations.append(relation_tag)
                    self.trigger_ids.append([self.trigger2id[x] for x in trigger_tags])
                    self.relation_ids.append(
                        [self.relation2id[x] for x in relation_tag]
                    )
                j += 1

        self.tokenized_inputs = self.tokenize_and_align_labels(
            self.tokens, self.trigger_ids, self.relation_ids
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_inputs["input_ids"][idx],
            "attention_mask": self.tokenized_inputs["attention_mask"][idx],
            "labels": self.tokenized_inputs["labels"][idx],
            "label_type_ids": self.tokenized_inputs["label_type_ids"][idx],
        }

    def tokenize_and_align_labels(self, tokens, tags_id, labels_type_id):
        tokenized_input = self.tokenizer(
            tokens, truncation=True, padding=True, is_split_into_words=True
        )

        tokenized_input["labels"] = self.align_labels(
            tags_id, tokenized_input, ignore_index=-100
        )
        # set 0 instead of -100 for implicit case
        tokenized_input["label_type_ids"] = self.align_labels(
            labels_type_id, tokenized_input, ignore_index=0 if self.implicit else -100
        )
        return tokenized_input

    def align_labels(self, tags_id, tokenized_input, ignore_index):
        """Taken from https://huggingface.co/docs/transformers/tasks/token_classification"""
        labels = []
        for i, label in enumerate(tags_id):
            word_ids = tokenized_input.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(ignore_index)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(ignore_index)
                previous_word_idx = word_idx
            labels.append(label_ids)
        return labels


class EDNYTDataset(data.Dataset):
    def __init__(self, fpath, tokenizer, trigger2id, task="trigger identification"):
        self.tokenizer = tokenizer
        self.trigger2id = trigger2id
        self.tokens, self.triggers, self.trigger_ids = [], [], []

        with open(fpath, "r") as f:
            data = json.load(f)
            for item in data:
                words = item["tokens"]
                tags = item["bio_tags"]

                self.tokens.append(words)
                self.triggers.append(tags)
                self.trigger_ids.append([self.trigger2id[x] for x in tags])

        self.tokenized_inputs = self.tokenize_and_align_labels(
            self.tokens, self.trigger_ids
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_inputs["input_ids"][idx],
            "attention_mask": self.tokenized_inputs["attention_mask"][idx],
            "labels": self.tokenized_inputs["labels"][idx],
        }

    def tokenize_and_align_labels(self, tokens, tags_id):
        """Taken from https://huggingface.co/docs/transformers/tasks/token_classification"""
        tokenized_input = self.tokenizer(
            tokens, truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(tags_id):
            word_ids = tokenized_input.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_input["labels"] = labels
        return tokenized_input


class EDNYTTriggerRelationDataset(data.Dataset):
    def __init__(
        self,
        fpath_trigger,
        fpath_relation,
        tokenizer,
        trigger2id,
        relation2id,
        implicit=False,
    ):
        self.tokenizer = tokenizer
        self.trigger2id = trigger2id
        self.relation2id = relation2id
        (
            self.tokens,
            self.triggers,
            self.relations,
            self.trigger_ids,
            self.relation_ids,
        ) = ([], [], [], [], [])

        # is the multitask learning implicit or explicit
        self.implicit = implicit

        visited = []

        with open(fpath_trigger, "r") as f:
            trigger_data = json.load(f)

        with open(fpath_relation, "r") as f:
            relation_data = json.load(f)

        for i, item in enumerate(trigger_data):
            tokens = item["tokens"]
            trigger_tags = item["bio_tags"]

            if str(i) in relation_data:
                relation_tags = relation_data[str(i)]["bio_tags"]
                # can be multiple bio_tags for one sentence or one
                for relation_tag in relation_tags:
                    # remove Subject and Object info
                    relation_tag = [
                        tag if tag in ["B-Relation", "I-Relation"] else OTHER
                        for tag in relation_tag
                    ]
                    # handle duplicates
                    triple = []
                    for token, t_tag, r_tag in zip(tokens, trigger_tags, relation_tag):
                        triple.append((token, t_tag, r_tag))
                    if triple not in visited:
                        self.tokens.append(tokens)
                        self.triggers.append(trigger_tags)
                        self.relations.append(relation_tag)
                        self.trigger_ids.append(
                            [self.trigger2id[x] for x in trigger_tags]
                        )
                        self.relation_ids.append(
                            [self.relation2id[x] for x in relation_tag]
                        )
            else:
                relation_tag = [OTHER] * len(tokens)
                self.tokens.append(tokens)
                self.triggers.append(trigger_tags)
                self.relations.append(relation_tag)
                self.trigger_ids.append([self.trigger2id[x] for x in trigger_tags])
                self.relation_ids.append([self.relation2id[x] for x in relation_tag])

        self.tokenized_inputs = self.tokenize_and_align_labels(
            self.tokens, self.trigger_ids, self.relation_ids
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_inputs["input_ids"][idx],
            "attention_mask": self.tokenized_inputs["attention_mask"][idx],
            "labels": self.tokenized_inputs["labels"][idx],
            "label_type_ids": self.tokenized_inputs["label_type_ids"][idx],
        }

    def tokenize_and_align_labels(self, tokens, tags_id, labels_type_id):
        tokenized_input = self.tokenizer(
            tokens,
            truncation=True,
            padding=True,
            is_split_into_words=True,
        )

        tokenized_input["labels"] = self.align_labels(
            tags_id, tokenized_input, ignore_index=-100
        )
        # set 0 instead of -100 for implicit case
        tokenized_input["label_type_ids"] = self.align_labels(
            labels_type_id, tokenized_input, ignore_index=0 if self.implicit else -100
        )
        return tokenized_input

    def align_labels(self, tags_id, tokenized_input, ignore_index):
        """Taken from https://huggingface.co/docs/transformers/tasks/token_classification"""
        labels = []
        for i, label in enumerate(tags_id):
            word_ids = tokenized_input.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(ignore_index)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(ignore_index)
                previous_word_idx = word_idx
            labels.append(label_ids)
        return labels


class EVEXTRADataset(data.Dataset):
    def __init__(self, fpath, tokenizer, trigger2id, task="trigger identification"):
        self.tokenizer = tokenizer
        self.trigger2id = trigger2id
        self.tokens, self.triggers, self.trigger_ids = [], [], []

        with open(fpath, "r") as f:
            data = json.load(f)
            for item in data:
                words = item["tokens"]
                tags = item["bio_tags"]

                self.tokens.append(words)
                self.triggers.append(tags)
                self.trigger_ids.append([self.trigger2id[x] for x in tags])

        self.tokenized_inputs = self.tokenize_and_align_labels(
            self.tokens, self.trigger_ids
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_inputs["input_ids"][idx],
            "attention_mask": self.tokenized_inputs["attention_mask"][idx],
            "labels": self.tokenized_inputs["labels"][idx],
        }

    def tokenize_and_align_labels(self, tokens, tags_id):
        """Taken from https://huggingface.co/docs/transformers/tasks/token_classification"""
        tokenized_input = self.tokenizer(
            tokens, truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(tags_id):
            word_ids = tokenized_input.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_input["labels"] = labels
        return tokenized_input


class EVEXTRATriggerRelationDataset(data.Dataset):
    def __init__(
        self,
        fpath_trigger,
        fpath_relation,
        tokenizer,
        trigger2id,
        relation2id,
        implicit=False,
    ):
        self.tokenizer = tokenizer
        self.trigger2id = trigger2id
        self.relation2id = relation2id
        (
            self.tokens,
            self.triggers,
            self.relations,
            self.trigger_ids,
            self.relation_ids,
        ) = ([], [], [], [], [])

        # is the multitask learning implicit or explicit
        self.implicit = implicit

        visited = []

        with open(fpath_trigger, "r") as f:
            trigger_data = json.load(f)

        with open(fpath_relation, "r") as f:
            relation_data = json.load(f)

        for i, item in enumerate(trigger_data):
            tokens = item["tokens"]
            trigger_tags = item["bio_tags"]

            if str(i) in relation_data:
                relation_tags = relation_data[str(i)]["bio_tags"]
                # can be multiple bio_tags for one sentence or one
                for relation_tag in relation_tags:
                    # remove Subject and Object info
                    relation_tag = [
                        tag if tag in ["B-Relation", "I-Relation"] else OTHER
                        for tag in relation_tag
                    ]
                    # handle duplicates
                    triple = []
                    for token, t_tag, r_tag in zip(tokens, trigger_tags, relation_tag):
                        triple.append((token, t_tag, r_tag))
                    if triple not in visited:
                        self.tokens.append(tokens)
                        self.triggers.append(trigger_tags)
                        self.relations.append(relation_tag)
                        self.trigger_ids.append(
                            [self.trigger2id[x] for x in trigger_tags]
                        )
                        self.relation_ids.append(
                            [self.relation2id[x] for x in relation_tag]
                        )
            else:
                relation_tag = [OTHER] * len(tokens)
                self.tokens.append(tokens)
                self.triggers.append(trigger_tags)
                self.relations.append(relation_tag)
                self.trigger_ids.append([self.trigger2id[x] for x in trigger_tags])
                self.relation_ids.append([self.relation2id[x] for x in relation_tag])

        self.tokenized_inputs = self.tokenize_and_align_labels(
            self.tokens, self.trigger_ids, self.relation_ids
        )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_inputs["input_ids"][idx],
            "attention_mask": self.tokenized_inputs["attention_mask"][idx],
            "labels": self.tokenized_inputs["labels"][idx],
            "label_type_ids": self.tokenized_inputs["label_type_ids"][idx],
        }

    def tokenize_and_align_labels(self, tokens, tags_id, labels_type_id):
        tokenized_input = self.tokenizer(
            tokens,
            truncation=True,
            padding=True,
            is_split_into_words=True,
        )

        tokenized_input["labels"] = self.align_labels(
            tags_id, tokenized_input, ignore_index=-100
        )
        # set 0 instead of -100 for implicit case
        tokenized_input["label_type_ids"] = self.align_labels(
            labels_type_id, tokenized_input, ignore_index=0 if self.implicit else -100
        )
        return tokenized_input

    def align_labels(self, tags_id, tokenized_input, ignore_index):
        """Taken from https://huggingface.co/docs/transformers/tasks/token_classification"""
        labels = []
        for i, label in enumerate(tags_id):
            word_ids = tokenized_input.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(ignore_index)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(ignore_index)
                previous_word_idx = word_idx
            labels.append(label_ids)
        return labels
