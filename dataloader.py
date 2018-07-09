import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dataset(Dataset):
    def __init__(self):
        self.data = pickle.load(open('./data/pcorpus', 'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    index = []
    max_len = 0
    for i in range(len(batch)):
        if max_len < len(batch[i]):
            max_len = len(batch[i])
        if batch[i][0] == 4:
            index.append(i)
            batch[i].pop(0)

    indices = sorted(range(len(batch)), key=[len(a) for a in batch].__getitem__, reverse=True)
    length = [len(batch[k]) for k in indices]
    src = [batch[k] + [0 for _ in range(max_len - len(batch[k]))] for k in indices]
    target = copy.copy(src)
    for idx in index:
        target[indices.index(idx)] = [4, 3] + [0 for _ in range(max_len - 2)]
    target = [[2] + k for k in target]

    return (torch.tensor(src, dtype=torch.int64, device=device),
            torch.tensor(target, dtype=torch.int64, device=device),
            torch.tensor(length, dtype=torch.int64, device=device))


def get_dataloader(batch_size, shuffle):
    return DataLoader(dataset(), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
