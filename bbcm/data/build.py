"""
@Time   :   2021-01-21 14:20:50
@File   :   build.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
from bbcm.utils import get_abs_path


def make_loaders(cfg, get_loader_fn, **kwargs):
    if cfg.DATASETS.TRAIN == '':
        train_loader = None
    else:
        train_loader = get_loader_fn(get_abs_path(cfg.DATASETS.TRAIN),
                                     batch_size=cfg.SOLVER.BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=cfg.DATALOADER.NUM_WORKERS, **kwargs)
    if cfg.DATASETS.VALID == '':
        valid_loader = None
    else:
        valid_loader = get_loader_fn(get_abs_path(cfg.DATASETS.VALID),
                                     batch_size=cfg.TEST.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=cfg.DATALOADER.NUM_WORKERS, **kwargs)
    if cfg.DATASETS.TEST == '':
        test_loader = None
    else:
        test_loader = get_loader_fn(get_abs_path(cfg.DATASETS.TEST),
                                    batch_size=cfg.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=cfg.DATALOADER.NUM_WORKERS, **kwargs)
    return train_loader, valid_loader, test_loader
