import torch

class DataCollactorForSpellGEC(object):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        src_tok = [self.tokenizer.tokenize(pair[0]) for pair in batch]
        trg_tok = [self.tokenizer.tokenize(pair[1]) for pair in batch]
        for s, t in zip(src_tok, trg_tok):
            assert len(s) == len(t), f"{s}   {t}"  # check if src length == trg length, after tokenize

        src = [pair[0] for pair in batch]
        src = self.tokenizer(src, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
        labels = []
        for i, l in enumerate(trg_tok):
            word_ids = src.word_ids(batch_index=i)
            label = []
            for word_id in word_ids:
                if word_id is None:
                    label.append(-100)
                else:
                    label.append(self.tokenizer.convert_tokens_to_ids(l[word_id]))
            labels.append(label)
        labels = torch.tensor(labels)
        return src, labels, src_tok, trg_tok
