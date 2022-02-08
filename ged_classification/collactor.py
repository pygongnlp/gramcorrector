import torch

class DataCollactorForGED(object):
    def __init__(self, tokenizer, max_length, label2id):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __call__(self, batch):
        inputs = [pair[0] for pair in batch]
        labels = [pair[1] for pair in batch]
        tok_inputs = self.tokenizer(inputs, return_tensors="pt",
                                    padding=True, truncation=True, max_length=self.max_length)
        labels = torch.tensor([self.label2id[label] for label in labels])
        return tok_inputs, labels, inputs
