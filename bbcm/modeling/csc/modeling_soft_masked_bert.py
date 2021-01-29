"""
@Time   :   2021-01-21 12:00:59
@File   :   modeling_soft_masked_bert.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import operator
import os
from collections import OrderedDict

import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertOnlyMLMHead
from transformers.modeling_utils import ModuleUtilsMixin
from bbcm.engine.csc_trainer import CscTrainingModel
import numpy as np


class DetectionNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            self.config.hidden_size,
            self.config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.config.hidden_dropout_prob,
            bidirectional=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(self.config.hidden_size, 1)

    def forward(self, hidden_states):
        out, _ = self.gru(hidden_states)
        prob = self.linear(out)
        prob = self.sigmoid(prob)
        return prob


class BertCorrectionModel(torch.nn.Module, ModuleUtilsMixin):
    def __init__(self, config, tokenizer, device):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embeddings = BertEmbeddings(self.config)
        self.corrector = BertEncoder(self.config)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pooler = BertPooler(self.config)
        self.cls = BertOnlyMLMHead(self.config)
        self._device = device

    def forward(self, texts, prob, embed=None, cor_labels=None, residual_connection=False):
        if cor_labels is not None:
            text_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')['input_ids']
            text_labels = text_labels.to(self._device)
            # torch的cross entropy loss 会忽略-100的label
            text_labels[text_labels == 0] = -100
        else:
            text_labels = None
        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_texts.to(self._device)
        if embed is None:
            embed = self.embeddings(input_ids=encoded_texts['input_ids'],
                                    token_type_ids=encoded_texts['token_type_ids'])
        # 此处较原文有一定改动，做此改动意在完整保留type_ids及position_ids的embedding。
        # mask_embed = self.embeddings(torch.ones_like(prob.squeeze(-1)).long() * self.mask_token_id).detach()
        # 此处为原文实现
        mask_embed = self.embeddings(torch.tensor([[self.mask_token_id]], device=self._device)).detach()
        cor_embed = prob * mask_embed + (1 - prob) * embed

        input_shape = encoded_texts['input_ids'].size()
        device = encoded_texts['input_ids'].device

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(encoded_texts['attention_mask'],
                                                                                 input_shape, device)
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        encoder_outputs = self.corrector(
            cor_embed,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        sequence_output = sequence_output + embed if residual_connection else sequence_output
        prediction_scores = self.cls(sequence_output)
        out = (prediction_scores, sequence_output, pooled_output)

        # Masked language modeling softmax layer
        if text_labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            cor_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), text_labels.view(-1))
            out = (cor_loss,) + out
        return out

    def load_from_transformers_state_dict(self, gen_fp):
        state_dict = OrderedDict()
        gen_state_dict = torch.load(gen_fp)
        for k, v in gen_state_dict.items():
            name = k
            if name.startswith('bert'):
                name = name[5:]
            if name.startswith('encoder'):
                name = f'corrector.{name[8:]}'
            if 'gamma' in name:
                name = name.replace('gamma', 'weight')
            if 'beta' in name:
                name = name.replace('beta', 'bias')
            state_dict[name] = v
        self.load_state_dict(state_dict, strict=False)


class SoftMaskedBertModel(CscTrainingModel):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.config = BertConfig.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.detector = DetectionNetwork(self.config)
        self.tokenizer = tokenizer
        self.corrector = BertCorrectionModel(self.config, tokenizer, cfg.MODEL.DEVICE)
        self._device = cfg.MODEL.DEVICE

    def forward(self, texts, cor_labels=None, det_labels=None):
        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_texts.to(self._device)
        embed = self.corrector.embeddings(input_ids=encoded_texts['input_ids'],
                                          token_type_ids=encoded_texts['token_type_ids'])
        prob = self.detector(embed)
        cor_out = self.corrector(texts, prob, embed, cor_labels, residual_connection=True)

        if det_labels is not None:
            det_loss_fct = nn.BCELoss()
            # pad部分不计算损失
            active_loss = encoded_texts['attention_mask'].view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())
            outputs = (det_loss, cor_out[0], prob.squeeze(-1)) + cor_out[1:]
        else:
            outputs = (prob.squeeze(-1),) + cor_out

        return outputs

    def load_from_transformers_state_dict(self, gen_fp):
        """
        从transformers加载预训练权重
        :param gen_fp:
        :return:
        """
        self.corrector.load_from_transformers_state_dict(gen_fp)
