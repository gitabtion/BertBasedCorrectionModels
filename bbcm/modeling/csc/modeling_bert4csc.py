"""
@Time   :   2021-01-22 11:42:52
@File   :   bert4csc.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""

import torch.nn as nn
from transformers import BertForMaskedLM

from bbcm.engine.csc_trainer import CscTrainingModel
from bbcm.solver.losses import FocalLoss


class BertForCsc(CscTrainingModel):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.bert = BertForMaskedLM.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.detection = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = tokenizer

    def forward(self, texts, cor_labels=None, det_labels=None):
        if cor_labels is not None:
            text_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')['input_ids']
            text_labels = text_labels.to(self.device)
            text_labels[text_labels == 0] = -100
        else:
            text_labels = None
        encoded_text = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_text.to(self.device)
        bert_outputs = self.bert(**encoded_text, labels=text_labels, return_dict=True, output_hidden_states=True)
        # 检错概率
        prob = self.detection(bert_outputs.hidden_states[-1])

        if text_labels is None:
            # 检错输出，纠错输出
            outputs = (prob, bert_outputs.logits)
        else:
            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')
            # pad部分不计算损失
            active_loss = encoded_text['attention_mask'].view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())
            # 检错loss，纠错loss，检错输出，纠错输出
            outputs = (det_loss,
                       bert_outputs.loss,
                       self.sigmoid(prob).squeeze(-1),
                       bert_outputs.logits)
        return outputs
