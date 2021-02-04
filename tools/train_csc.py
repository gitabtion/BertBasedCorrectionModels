"""
@Time   :   2021-01-21 11:47:09
@File   :   train_csc.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import sys
sys.path.append('..')


from pytorch_lightning.callbacks import ModelCheckpoint
from bbcm.data.build import make_loaders
from bbcm.data.loaders import get_csc_loader
from bbcm.modeling.csc import SoftMaskedBertModel
from bbcm.modeling.csc.modeling_bert4csc import BertForCsc
from transformers import BertTokenizer
from bases import args_parse, train
from bbcm.utils import get_abs_path
from bbcm.data.processors.csc import preproc
import os


def main():
    cfg = args_parse("csc/train_macbert4csc.yml")

    # 如果不存在训练文件则先处理数据
    if not os.path.exists(get_abs_path(cfg.DATASETS.TRAIN)):
        preproc()
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
    if cfg.MODEL.NAME in ["bert4csc", "macbert4csc"]:
        model = BertForCsc(cfg, tokenizer)
    else:
        model = SoftMaskedBertModel(cfg, tokenizer)

    if len(cfg.MODEL.WEIGHTS) > 0:
        ckpt_path = get_abs_path(cfg.OUTPUT_DIR, cfg.MODEL.WEIGHTS)
        model.load_from_checkpoint(ckpt_path, cfg=cfg, tokenizer=tokenizer)

    loaders = make_loaders(cfg, get_csc_loader, tokenizer=tokenizer)
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=get_abs_path(cfg.OUTPUT_DIR),
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    train(cfg, model, loaders, ckpt_callback)


if __name__ == '__main__':
    main()
