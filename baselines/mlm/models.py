# author: ddukic

import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import (
    MaskedLMOutput,
    RobertaForTokenClassification,
    RobertaLMHead,
    TokenClassifierOutput,
)


class RobertaForMLMTriggerTokenClassification(RobertaForTokenClassification):
    # adapted from https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/roberta/modeling_roberta.py
    def __init__(self, config):
        super().__init__(config=config)

        self.lm_head = RobertaLMHead(config)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        label_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # get last hidden state
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_mlm(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        return MaskedLMOutput(
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMLMTriggerRelationTokenClassification(RobertaForTokenClassification):
    # adapted from https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/roberta/modeling_roberta.py
    def __init__(self, config, label_type_size, label_hidden_size):
        super().__init__(config=config)

        self.label_type_embeddings = nn.Embedding(label_type_size, label_hidden_size)
        self.classifier = nn.Linear(
            config.hidden_size + label_hidden_size, config.num_labels
        )

        self.lm_head = RobertaLMHead(config)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        label_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # get last hidden state
        sequence_output = outputs[0]

        # to appply dropout before or after concatenating?
        # sequence_output = self.dropout(sequence_output)

        label_emb_indexed = self.label_type_embeddings(label_type_ids)

        sequence_output = torch.cat(
            [
                sequence_output,
                label_emb_indexed,
            ],
            dim=-1,
        )

        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_mlm(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        return MaskedLMOutput(
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMLMMultiTaskTwoHeadsTokenClassification(RobertaForTokenClassification):
    # adapted from https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/roberta/modeling_roberta.py
    def __init__(self, config, id2labels):
        super().__init__(config=config)

        self.heads = nn.ModuleDict()
        for k in id2labels.keys():
            self.heads[k] = nn.Linear(config.hidden_size, self.num_labels)

        self.id2labels = id2labels

        self.lm_head = RobertaLMHead(config)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels_key=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # get last hidden state
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        # pass through trigger head or relation head
        logits = self.heads[labels_key](sequence_output)

        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_mlm(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        return MaskedLMOutput(
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
