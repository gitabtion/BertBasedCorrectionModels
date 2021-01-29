"""
@Time   :   2021-01-21 11:24:00
@File   :   csc.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
from torch.utils.data import Dataset

from bbcm.utils import load_json


class CscDataset(Dataset):
    def __init__(self, fp):
        self.data = load_json(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['original_text'], self.data[index]['correct_text'], self.data[index]['wrong_ids']
