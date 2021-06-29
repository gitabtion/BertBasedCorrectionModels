# BertBasedCorrectionModels

基于BERT的文本纠错模型，使用PyTorch实现

## 数据准备
1. 从 [http://nlp.ee.ncu.edu.tw/resource/csc.html](http://nlp.ee.ncu.edu.tw/resource/csc.html)下载SIGHAN数据集
2. 解压上述数据集并将文件夹中所有 ''.sgml'' 文件复制至 datasets/csc/ 目录
3. 复制 ''SIGHAN15_CSC_TestInput.txt'' 和 ''SIGHAN15_CSC_TestTruth.txt'' 至 datasets/csc/ 目录
4. 下载 [https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml) 至 datasets/csc 目录
5. 请确保以下文件在 datasets/csc 中
    ```
    train.sgml
    B1_training.sgml
    C1_training.sgml  
    SIGHAN15_CSC_A2_Training.sgml  
    SIGHAN15_CSC_B2_Training.sgml  
    SIGHAN15_CSC_TestInput.txt
    SIGHAN15_CSC_TestTruth.txt
    ```

## 环境准备
1. 使用已有编码环境或通过 `conda create -n <your_env_name> python=3.7` 创建一个新环境（推荐）
2. 克隆本项目并进入项目根目录 
3. 安装所需依赖 `pip install -r requirements.txt`
4. 如果出现报错 GLIBC 版本过低的问题（GLIBC 的版本更迭容易出事故，不推荐更新），openCC 改为安装较低版本（例如 1.1.0）
5. 在当前终端将此目录加入环境变量 `export PYTHONPATH=.`


## 训练

运行以下命令以训练模型，首次运行会自动处理数据。
```shell
python tools/train_csc.py --config_file csc/train_SoftMaskedBert.yml
```

可选择不同配置文件以训练不同模型，目前支持以下配置文件：
- train_bert4csc.yml
- train_macbert4csc.yml
- train_SoftMaskedBert.yml

如有其他需求，可根据需要自行调整配置文件中的参数。

## 实验结果

### SoftMaskedBert
|component|sentence level acc|p|r|f|
|:-:|:-:|:-:|:-:|:-:|
|Detection|0.5045|0.8252|0.8416|0.8333|
|Correction|0.8055|0.9395|0.8748|0.9060|

### Bert类
#### char level
|MODEL|p|r|f|
|:-:|:-:|:-:|:-:|
|BERT4CSC|0.9269|0.8651|0.8949|
|MACBERT4CSC|0.9380|0.8736|0.9047|

#### sentence level
|model|acc|p|r|f|
|:-:|:-:|:-:|:-:|:-:|
|BERT4CSC|0.7990|0.8482|0.7214|0.7797|
|MACBERT4CSC|0.8027|0.8525|0.7251|0.7836|

## 推理
### 方法一，使用inference脚本:
```shell
python inference.py --ckpt_fn epoch=0-val_loss=0.03.ckpt --texts "我今天很高心"
# 或给出line by line格式的文本地址
python inference.py --ckpt_fn epoch=0-val_loss=0.03.ckpt --text_file /ml/data/text.txt
```
其中/ml/data/text.txt文本如下：
```text
我今天很高心
你这个辣鸡模型只能做错别字纠正
```
### 方法二，直接调用
```python
from tools.inference import *
ckpt_fn = 'SoftMaskedBert/epoch=02-val_loss=0.02904.ckpt'  # find it in checkpoints/
config_file = 'csc/train_SoftMaskedBert.yml'  # find it in configs/
model = load_model_directly(ckpt_fn=ckpt_fn, config_file=config_file)
texts = ['今天我很高心', '测试', '继续测试']
model.predict(texts)
```
### 方法三、导出bert权重，使用transformers或pycorrector调用
1. 使用convert_to_pure_state_dict.py导出bert权重
2. 后续步骤参考[https://github.com/shibing624/pycorrector/blob/master/pycorrector/macbert/README.md](https://github.com/shibing624/pycorrector/blob/master/pycorrector/macbert/README.md)

## 引用
如果你在研究中使用了本项目，请按如下格式引用：

```
@article{cai2020pre,
  title={BERT Based Correction Models},
  author={Cai, Heng and Chen, Dian},
  journal={GitHub. Note: https://github.com/gitabtion/BertBasedCorrectionModels},
  year={2020}
}
```

## License
本源代码的授权协议为 Apache License 2.0，可免费用做商业用途。请在产品说明中附加本项目的链接和授权协议。本项目受版权法保护，侵权必究。


## 更新记录

### 20210618
1. 修复数据处理的编码报错问题

### 20210518
1. 将BERT4CSC检错任务改为使用FocalLoss
2. 更新修改后的模型实验结果
3. 降低数据处理时保留原文的概率

### 20210517
1. 对BERT4CSC模型新增检错任务
2. 新增基于LineByLine文件的inference

## References
1. [Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/abs/2005.07421)
2. [http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html](http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html)
3. [https://github.com/wdimmy/Automatic-Corpus-Generation](https://github.com/wdimmy/Automatic-Corpus-Generation)
4. [transformers](https://huggingface.co/)
5. [https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check](https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check)
6. [SoftMaskedBert-PyTorch](https://github.com/gitabtion/SoftMaskedBert-PyTorch)
7. [Deep-Learning-Project-Template](https://github.com/L1aoXingyu/Deep-Learning-Project-Template)
8. [https://github.com/lonePatient/TorchBlocks](https://github.com/lonePatient/TorchBlocks)
9. [https://github.com/shibing624/pycorrector](https://github.com/shibing624/pycorrector)
