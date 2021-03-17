"""
@Time   :   2021-01-21 11:17:25
@File   :   bases.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import argparse
import os

import logging

import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from bbcm.utils import get_abs_path
from bbcm.utils.logger import setup_logger
from bbcm.config import cfg
import pytorch_lightning as pl
import os


def args_parse(config_file=''):
    parser = argparse.ArgumentParser(description="bbcm")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("--opts", help="Modify config options using the command-line key value", default=[],
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    config_file = args.config_file or config_file

    if config_file != "":
        cfg.merge_from_file(get_abs_path('configs', config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    name = cfg.MODEL.NAME

    output_dir = cfg.OUTPUT_DIR

    logger = setup_logger(name, get_abs_path(output_dir), 0)
    logger.info(args)

    if config_file != '':
        logger.info("Loaded configuration file {}".format(config_file))
        with open(get_abs_path('configs', config_file), 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    return cfg


def train(config, model, loaders, ckpt_callback=None):
    """
    训练
    Args:
        config: 配置
        model: 模型
        loaders: 各个数据的loader，包含train，valid，test
        ckpt_callback: 按需保存模型的callback，如为空则默认每个epoch保存一次模型。
    Returns:
        None
    """
    train_loader, valid_loader, test_loader = loaders
    trainer = pl.Trainer(max_epochs=config.SOLVER.MAX_EPOCHS,
                         gpus=None if config.MODEL.DEVICE == 'cpu' else config.MODEL.GPU_IDS,
                         accumulate_grad_batches=config.SOLVER.ACCUMULATE_GRAD_BATCHES,
                         checkpoint_callback=ckpt_callback)
    # 满足以下条件才进行训练
    # 1. 配置文件中要求进行训练
    # 2. train_loader不为空
    # 3. train_loader中有数据
    if 'train' in config.MODE and train_loader and len(train_loader) > 0:
        if valid_loader and len(valid_loader) > 0:
            trainer.fit(model, train_loader, valid_loader)
        else:
            trainer.fit(model, train_loader)
    # 是否进行测试的逻辑同训练
    if 'test' in config.MODE and test_loader and len(test_loader) > 0:
        if ckpt_callback and len(ckpt_callback.best_model_path) > 0:
            ckpt_path = ckpt_callback.best_model_path
        elif len(config.MODEL.WEIGHTS) > 0:
            ckpt_path = get_abs_path(config.OUTPUT_DIR, config.MODEL.WEIGHTS)
        else:
            ckpt_path = None
        print(ckpt_path)
        if (ckpt_path is not None) and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        trainer.test(model, test_loader)
