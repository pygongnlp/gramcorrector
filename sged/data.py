from torch.utils.data import Dataset
from utils import load_data, get_labels

class GEDDataset(Dataset):
    def __init__(self, file_path, mode):
        src_lst, trg_lst = load_data(file_path, mode)
        self.src_lst = src_lst
        self.trg_lst = trg_lst
        self.labels = get_labels(src_lst, trg_lst)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.src_lst[item], self.trg_lst[item], self.labels[item]


