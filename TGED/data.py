from torch.utils.data import Dataset
from utils import load_data

class SpellGECDataset(Dataset):
    def __init__(self, file_path, mode):
        self.src_lst, self.trg_lst = load_data(file_path, mode)

    def __len__(self):
        return len(self.src_lst)

    def __getitem__(self, item):
        return self.src_lst[item], self.trg_lst[item]