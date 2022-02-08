from torch.utils.data import Dataset
from utils import load_data

class GEDDataset(Dataset):
    def __init__(self, file_path, mode):
        self.sents, self.labels = load_data(file_path, mode)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.sents[item], self.labels[item]