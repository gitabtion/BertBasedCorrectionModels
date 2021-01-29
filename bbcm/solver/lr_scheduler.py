"""
@Time   :   2021-01-21 10:52:47
@File   :   lr_scheduler.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
from torch.optim.lr_scheduler import LambdaLR


def make_scheduler(cfg, optimizer):
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda step: min((step + 1) ** -0.5,
                                                    (step + 1) * cfg.SOLVER.WARMUP_EPOCHS ** (-1.5)),
                         last_epoch=-1)
    return scheduler
