import torch

class DataCollactorForTGED(object):
    def __init__(self, tokenizer, max_length, label2id):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __call__(self, batch):
        src_oral = [pair[0] for pair in batch]
        trg_oral = [pair[1] for pair in batch]
        label_lst = [pair[2] for pair in batch]

        src = self.tokenizer(src_oral, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
        labels = []
        for i, l in enumerate(label_lst):
            word_ids = src.word_ids(batch_index=i)
            label = []
            for word_id in word_ids:
                if word_id is None:
                    label.append(-100)
                else:
                    label.append(self.label2id[l[word_id]])
            labels.append(label)
        labels = torch.tensor(labels)
        return src, labels, src_oral, trg_oral
