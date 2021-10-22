import unittest

import transformers as tfs


class TestEmbedding(unittest.TestCase):
    def test_embedding(self):
        model = tfs.AutoModel.from_pretrained('bert-base-chinese')
        tokenizer = tfs.AutoTokenizer.from_pretrained('bert-base-chinese')
       