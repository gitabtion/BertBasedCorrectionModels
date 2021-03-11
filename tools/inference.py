"""
@Time   :   2021-02-05 15:33:55
@File   :   inference.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import sys
import argparse
import os

sys.path.append('..')
from bbcm.modeling.csc import BertForCsc, SoftMaskedBertModel
from bbcm.utils import get_abs_path


def parse_args():
    parser = argparse.ArgumentParser(description="bbcm")
    parser.add_argument(
        "--model_name", default="bert4csc", help="model name", type=str
    )
    parser.add_argument(
        "--ckpt_fn", default="", help="checkpoint file name", type=str
    )
    parser.add_argument("--texts", default=["马上要过年了，提前祝大家心年快乐！"], nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def load_model(args):
    file_dir = get_abs_path("checkpoints", args.model_name)
    if args.model_name in ['bert4csc', 'macbert4csc']:
        model = BertForCsc.load_from_checkpoint(os.path.join(file_dir, args.ckpt_fn))
    else:
        model = SoftMaskedBertModel.load_from_checkpoint(os.path.join(file_dir, args.ckpt_fn))

    return model


def inference(args):
    model = load_model(args)
    corrected_texts = model.predict(args.texts)
    print(corrected_texts)
    return corrected_texts


if __name__ == '__main__':
    arguments = parse_args()
    inference(arguments)
