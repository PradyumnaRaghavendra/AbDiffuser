import pickle
import torch
from torch.utils.data import Dataset

class ProcessedOASDataset(Dataset):
    def __init__(self, file_path, max_examples=None):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        if max_examples:
            self.data = self.data[:max_examples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'one_hot': self.data[idx]['one_hot'],  # (seq_len, 21) numpy
            'positions': self.data[idx]['positions'],  # (seq_len, 4, 3) numpy
        }

    @staticmethod
    def collate_fn(batch):
        one_hot = [torch.tensor(item['one_hot']).float() for item in batch]  # <--- ADD .float()
        positions = [torch.tensor(item['positions']).float() for item in batch]  # <--- ADD .float()
        return {
            'one_hot': torch.stack(one_hot),
            'positions': torch.stack(positions),
        }
