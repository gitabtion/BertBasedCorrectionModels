"""
@Time   :   2021-01-22 11:42:52
@File   :   bert4csc.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""

import torch
from transformers import BertForMaskedLM

from bbcm.engine.csc_trainer import CscTrainingModel


class BertForCsc(CscTrainingModel):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.bert = BertForMaskedLM.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.tokenizer = tokenizer

    def forward(self, texts, cor_labels=None, det_labels=None):
        if cor_labels is not None:
            text_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')['input_ids']
            text_labels = text_labels.to(self.cfg.MODEL.DEVICE)
            text_labels[text_labels == 0] = -100
        else:
            text_labels = None
        encoded_text = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_text.to(self.cfg.MODEL.DEVICE)
        bert_outputs = self.bert(**encoded_text, labels=text_labels, return_dict=False)

        if text_labels is None:
            outputs = (torch.zeros_like(encoded_text['input_ids']),) + bert_outputs
        else:
            outputs = (torch.tensor(0.0, dtype=torch.float),
                       bert_outputs[0],
                       det_labels,) + bert_outputs[1:]
        return outputs
