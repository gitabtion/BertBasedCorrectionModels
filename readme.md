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

## 训练

运行以下命令以训练模型，首次运行会自动处理数据。
```shell
python tools/train_csc.py --config_file train_SoftMaskedBert.yml
```
可选择不同配置文件以训练不同模型，目前支持以下配置文件：
- train_bert4csc.yml
- train_macbert4csc.yml
- train_SoftMaskedBert.yml

## 实验结果
|component|sentence level acc|p|r|f|
|:-:|:-:|:-:|:-:|:-:|
|SoftMaskedBert Detection|0.5045|0.8252|0.8416|0.8333|
|SoftMaskedBert Correction|0.8055|0.9395|0.8748|0.9060|

## References
1. [Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/abs/2005.07421)
2. [http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html](http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html)
3. [https://github.com/wdimmy/Automatic-Corpus-Generation](https://github.com/wdimmy/Automatic-Corpus-Generation)
4. [transformers](https://huggingface.co/)
5. [https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check](https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check)
6. [SoftMaskedBert-PyTorch](https://github.com/gitabtion/SoftMaskedBert-PyTorch)
7. [Deep-Learning-Project-Template](https://github.com/L1aoXingyu/Deep-Learning-Project-Template)
