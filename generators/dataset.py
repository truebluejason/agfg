import torch
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, df: DataFrame):
        self.X = df

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X.iloc[idx].to_numpy()).to(torch.float32)
