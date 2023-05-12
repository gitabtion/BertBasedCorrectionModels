"""
@Time   :   2021-02-05 15:33:55
@File   :   inference.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import sys
import argparse
import os
import torch
from transformers import BertTokenizer
from tools.bases import args_parse
sys.path.append('..')
from bbcm.modeling.csc import BertForCsc, SoftMaskedBertModel
from bbcm.utils import get_abs_path
import json
import codecs
import re

def parse_args():
    parser = argparse.ArgumentParser(description="bbcm")
    parser.add_argument(
        "--config_file", default="csc/train_bert4csc.yml", help="config file", type=str
    )
    parser.add_argument(
        "--ckpt_fn", default="epoch=2-val_loss=0.02.ckpt", help="checkpoint file name", type=str
    )
    parser.add_argument("--texts", default=["马上要过年了，提前祝大家心年快乐！"], nargs=argparse.REMAINDER)
    parser.add_argument("--text_file", default='')

    args = parser.parse_args()
    return args


def load_model_directly(ckpt_file, config_file):
    # Example:
    # ckpt_fn = 'SoftMaskedBert/epoch=02-val_loss=0.02904.ckpt' (find in checkpoints)
    # config_file = 'csc/train_SoftMaskedBert.yml' (find in configs)
    
    from bbcm.config import cfg
    cp = get_abs_path('checkpoints', ckpt_file)
    cfg.merge_from_file(get_abs_path('configs', config_file))
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)

    if cfg.MODEL.NAME in ['bert4csc', 'macbert4csc']:
        model = BertForCsc.load_from_checkpoint(cp,
                                                cfg=cfg,
                                                tokenizer=tokenizer)
    else:
        model = SoftMaskedBertModel.load_from_checkpoint(cp,
                                                         cfg=cfg,
                                                         tokenizer=tokenizer)
    model.eval()
    model.to(cfg.MODEL.DEVICE)
    return model


def load_model(args):
    from bbcm.config import cfg
    cfg.merge_from_file(get_abs_path('configs', args.config_file))
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
    file_dir = get_abs_path("checkpoints", cfg.MODEL.NAME)
    if cfg.MODEL.NAME in ['bert4csc', 'macbert4csc']:
        model = BertForCsc.load_from_checkpoint(os.path.join(file_dir, args.ckpt_fn),
                                                cfg=cfg,
                                                tokenizer=tokenizer)
    else:
        model = SoftMaskedBertModel.load_from_checkpoint(os.path.join(file_dir, args.ckpt_fn),
                                                         cfg=cfg,
                                                         tokenizer=tokenizer)
    model.eval()
    model.to(cfg.MODEL.DEVICE)

    return model


def inference(args):
    model = load_model(args)
    texts = []
    if os.path.exists(args.text_file):
        with open(args.text_file, 'r', encoding='utf-8') as f:
            for line in f:
                texts.append(line.strip())
    else:
        texts = args.texts
    print("传入 的原始文本:{}".format(texts))
    corrected_texts = model.predict(texts)  # input is list and output is list
    print("模型纠错输出文本:{}".format(corrected_texts))
    # 输出结果后处理模块
    corrected_info = output_result(corrected_texts, sources=texts)
    print("模型纠错字段信息:{}".format(corrected_info))
    return corrected_texts

def parse_args_test():
    parser = argparse.ArgumentParser(description="bbcm")
    parser.add_argument(
        "--config_file", default="csc/train_SoftMaskedBert.yml", help="config file", type=str
    )
    parser.add_argument(
        "--ckpt_fn", default="epoch=09-val_loss=0.03032.ckpt", help="checkpoint file name", type=str
    )
    args = parser.parse_args()
    return args


def inference_test(texts):
    """input is texts list"""
    # 加载推理模型
    args = parse_args_test()
    # 加载模型参数
    model = load_model(args)
    #print("传入 的原始文本:{}".format(texts))
    corrected_texts = model.predict(texts)  # input is list and output is list
    #print("模型纠错输出文本:{}".format(corrected_texts))
    # 输出结果后处理模块
    corrected_info = output_result(corrected_texts, sources=texts)
    #print("模型纠错字段信息:{}".format(corrected_info))
    return corrected_texts, corrected_info


def load_json(filename, encoding="utf-8"):
    """Load json file"""
    if not os.path.exists(filename):
        return None
    with codecs.open(filename, mode='r', encoding=encoding) as fr:
        return json.load(fr)


# 预先加载 - 白名单 - 可根据实际应用场景定向更新后放入此推理代码中备用
white_dict = load_json("../configs/dict/white_name_list.json")  # 注意这里的路径-否则white_dict is None
# 编译中文字符
re_han = re.compile("[\u4E00-\u9Fa5]+")


def load_white_dict():
    default_lens = 4  # 根据配置的过纠字对应的语义片段长度来设定。默认值，可修改
    lens_list = list()
    for src in white_dict.keys():
        for name in white_dict[src]:
            lens_list.append(len(name))
    max_lens = max(lens_list) if lens_list else default_lens
    return white_dict, max_lens


def output_result(results, sources):
    """
    :param results:  模型纠错结果list
    :param sources:  输入list
    :return:
    """
    """封装输出格式"""
    default_data = [
        {
            "src_sentence": "",
            "tgt_sentence": "",
            "fragments": []
        }
    ]
    if not results:
        return default_data
    data = []
    # 一个result 生成一个字典dict()
    for idx, result in enumerate(results):
        # 源文本
        source = sources[idx]
        # 找到diff_info不同的地方
        fragments_lst = generate_diff_info(source, result)
        dict_res = {
            "src_sentence": source,
            "tgt_sentence": result,
            "fragments": fragments_lst
        }
        data.append(dict_res)
    return data


def generate_diff_info(source, result):
    """
    :param source: 原始输入文本 string
    :param result: 纠错模型输出文本 string
    :return: fragments, 输出[dict_1, dict_2, ....], dict_i 是每个字的纠错输出信息
    """
    """基于原始输入文本和纠错后的文本输出differ_info"""
    # 定义默认输出
    fragments = list()
    # 仅支持输出和输出相同的情况下，如果不同则fragments输出为空
    # 后处理逻辑1
    if len(source) != len(result):
        return fragments
    # 后处理逻辑2 - 如果输入的source中没有或仅有一个中文字符则也不处理
    res_hans = re_han.findall(source)
    if not res_hans:
        return fragments
    if res_hans and len(res_hans[0]) < 2:
        return fragments
    # 后处理逻辑3 - 逐个字段比对，输出不同的字的位置
    for idx in range(len(source)):
        # 原始字
        src = source[idx]
        # 模型输出的字
        tgt = result[idx]
        # 如果字没发生变化则按照没有错误处理
        if src == tgt:
            continue
        # 过滤掉非汉字
        if not re_han.findall(src):
            continue
        # 通过白名单过滤掉overcorrection-误杀的情况
        if model_white_list_filter(source, src, idx):
            continue

        # 找到不同的字所在index
        fragment = {
            "error_init_id": idx,  # 出错字开始位置索引
            "error_end_id": idx + 1,  # 结束索引
            "src_fragment": src,  # 原字
            "tgt_fragment": tgt  # 纠正后的字
        }
        fragments.append(fragment)
    return fragments


def model_white_list_filter(source, src, src_idx):
    """"source: 原来的句子; texts: 白名单; rules: 白名单规则"""
    """模型输出结果白名单过滤"""
    is_correct = False
    # 加载白名单
    wh_texts, span_w = load_white_dict()
    source_lens = len(source)
    if src in wh_texts.keys():
        for src_span in wh_texts[src]:
            # 如果配置的语义片段src_span在 传入的文本text 片段source[span_start:span_end]中，则认为过纠is_correct is True。
            span_start = src_idx-span_w
            span_end = src_idx+span_w
            span_start = 0 if span_start < 0 else span_start
            span_end = span_end if span_end < source_lens else source_lens
            if src_span in source[span_start:span_end]:
                is_correct = True
                return is_correct
    return is_correct


if __name__ == '__main__':
    # 原来推理代码
    # arguments = parse_args()
    # inference(arguments)
    # 添加代码后的测试代码如下:
    texts = [
        '真麻烦你了。希望你们好好的跳无',
        '少先队员因该为老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '今天心情很好',
        '汽车新式在这条路上',
        '中国人工只能布局很不错'
    ]
    corrected_texts, corrected_info = inference_test(texts)
    for info in corrected_info:
        print("----------------------")
        print("info:{}".format(info))
