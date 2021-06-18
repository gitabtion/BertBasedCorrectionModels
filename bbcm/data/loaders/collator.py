import torch


class DataCollatorForCsc:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        ori_texts, cor_texts, wrong_idss = zip(*data)
        encoded_texts = [self.tokenizer.tokenize(t) for t in ori_texts]
        max_len = max([len(t) for t in encoded_texts]) + 2
        det_labels = torch.zeros(len(ori_texts), max_len).long()
        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            for idx in wrong_ids:
                margins = []
                for word in encoded_text[:idx]:
                    if word == '[UNK]':
                        break
                    if word.startswith('##'):
                        margins.append(len(word) - 3)
                    else:
                        margins.append(len(word) - 1)
                margin = sum(margins)
                move = 0
                while (abs(move) < margin) or (idx + move >= len(encoded_text)) or encoded_text[idx + move].startswith(
                        '##'):
                    move -= 1
                det_labels[i, idx + move + 1] = 1
        return ori_texts, cor_texts, det_labels
