import torch

class DataCollactorForSGED(object):
    def __init__(self, tokenizer, max_length, label2id):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __call__(self, batch):
        src_oral = [pair[0] for pair in batch]
        src = self.tokenizer(src_oral, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
        trg_oral = [pair[1] for pair in batch]
        labels = [self.label2id[pair[2]] for pair in batch]
        labels = torch.tensor(labels)
        return src, labels, src_oral, trg_oral
