from torch.utils.data import Dataset


class SpellGECDataset(Dataset):
    def __init__(self, src_lst, trg_lst):
        self.src_lst = src_lst
        self.trg_lst = trg_lst

    def __len__(self):
        return len(self.src_lst)

    def __getitem__(self, item):
        return self.src_lst[item], self.trg_lst[item]