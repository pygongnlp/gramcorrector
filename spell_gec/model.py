import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import TokenClassifierOutput


class SpellGEC(nn.Module):

    def __init__(self, pretrained_model, num_labels, hidden_size, dropout=0.1):
        super(SpellGEC, self).__init__()
        self.num_labels = num_labels

        self.pretrained_model = pretrained_model

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, x, labels):

        outputs = self.pretrained_model(**x)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = self.fc(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )