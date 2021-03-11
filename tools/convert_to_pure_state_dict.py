"""
@Time   :   2021-02-03 20:58:46
@File   :   convert_to_pure_state_dict.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import sys
import argparse
import os
import torch
from collections import OrderedDict

sys.path.append('..')
from bbcm.utils import get_abs_path


def convert(fn, model_name):
    """
    从保存的ckpt文件中取出模型的state_dict用于迁移。
    Args:
        fn: ckpt文件的文件名
        model_name: 模型名，应与yml中的一致。

    Returns:

    """
    file_dir = get_abs_path("checkpoints", model_name)
    state_dict = torch.load((os.path.join(file_dir, fn)))['state_dict']
    new_state_dict = OrderedDict()
    if model_name in ['bert4csc', 'macbert4csc']:
        for k, v in state_dict.items():
            new_state_dict[k[5:]] = v
    else:
        new_state_dict = state_dict
    torch.save(new_state_dict, os.path.join(file_dir, 'pytorch_model.bin'))


def parse_args():
    parser = argparse.ArgumentParser(description="fast-bbdl")
    parser.add_argument(
        "--ckpt_fn", default="", help="checkpoint file name", type=str
    )
    parser.add_argument(
        "--model_name", default="bert4csc", help="model name, candidates: bert4csc, macbert4csc, SoftMaskedBert", type=str
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arguments = parse_args()
    convert(arguments.ckpt_fn, arguments.model_name)
