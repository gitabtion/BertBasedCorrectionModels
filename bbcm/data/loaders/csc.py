"""
@Time   :   2021-01-21 14:58:30
@File   :   csc.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
from torch.utils.data import DataLoader

from bbcm.data.datasets.csc import CscDataset


def get_csc_loader(fp, _collate_fn, **kwargs):
    dataset = CscDataset(fp)
    loader = DataLoader(dataset, collate_fn=_collate_fn, **kwargs)
    return loader
