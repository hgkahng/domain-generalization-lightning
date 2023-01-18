
import torch
import torch.nn as nn

from transformers import DistilBertModel
from dg_lightning.networks.base import _BertBackboneBase


class DistilBertBackbone(_BertBackboneBase):
    def __init__(self, name: str = 'distilbert-base-uncased'):
        """
        Borrows implementation from `transformers.DistilBertForSequenceClassification`,
            except the classification layer.
        """
        super(DistilBertBackbone, self).__init__()
        
        self.name: str = name
        self.distilbert = DistilBertModel.from_pretrained(self.name)
        self.pre_classifer = nn.Linear(
            in_features=self.distilbert.config.dim,   # 768
            out_features=self.distilbert.config.dim,  # 768
            bias=True,
        )
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=self.distilbert.config.seq_classif_dropout)

        self._init_weights()
    
    def forward(self, x: torch.LongTensor):
        """
        Arguments:
            x: 3d LongTensor of shape (B, max_length, 2)
        """
        model_input = {
            'input_ids': x[:, :, 0],
            'attention_mask': x[:, :, 1],
        }
        last_hidden_states = self.distilbert(**model_input)[0]  # (B, max_length, 768)
        pooled_output = last_hidden_states[:, 0]                # (B, 768)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        return pooled_output

    def _init_weights(self):
        """
        Initializes weights of immediate children. Implementation based on
            https://github.com/huggingface/transformers/blob/d719bcd46a70302486f2255f1ff7759e232a0122/src/transformers/models/distilbert/modeling_distilbert.py#L379
        """
        for _, child in self.named_children():
            if isinstance(child, nn.Linear):
                child.weight.data.normal_(mean=0., std=self.distilbert.config.initializer_range)
                if child.bias is not None:
                    child.bias.data.zero_()
            elif isinstance(child, nn.Embedding):
                child.weight.data.normal_(mean=0., std=self.distilbert.config.initializer_range)
                if child.padding_idx is not None:
                    child.weight.data[child.padding_idx].zero_()
            elif isinstance(child, nn.LayerNorm):
                child.bias.data.zero_()
                child.weight.data.fill_(1.)
        
    @property
    def out_features(self) -> int:
        return self.distilbert.config.dim  # 768
