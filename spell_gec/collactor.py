import torch

class DataCollactorForSpellGEC(object):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        src_oral = [self.tokenizer.tokenize(pair[0]) for pair in batch]
        trg_oral = [self.tokenizer.tokenize(pair[1]) for pair in batch]
        for s, t in zip(src_oral, trg_oral):
            assert len(s) == len(t), f"{s}   {t}"  # check if src length == trg length, after tokenize

        src = [pair[0] for pair in batch]
        src = self.tokenizer(src, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
        trg = []
        for i, label in enumerate(trg_oral):
            word_ids = src.word_ids(batch_index=i)
            t = []
            for word_id in word_ids:
                if word_id is None:
                    t.append(-100)
                else:
                    t.append(self.tokenizer.convert_tokens_to_ids(label[word_id]))
            trg.append(t)
        trg = torch.tensor(trg)
        return src, trg, src_oral, trg_oral
