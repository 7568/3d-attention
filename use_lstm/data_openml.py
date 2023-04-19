import torch
from torch.utils.data import Dataset
import numpy as np

class DataSetCatCon(Dataset):
    def __init__(self, data,sequence_length,features_n):
        target_fea = 'ClosePrice'
        # print(data.columns.to_numpy()[::-1].reshape(5,6))
        data = data[data.columns.to_numpy()[::-1]].to_numpy()
        train_y = data[:,-1].copy()
        data[:,-1]=-1


        self.x = data.reshape(-1,sequence_length,features_n)
        self.y = train_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
