import torch
from torch.utils.data import Dataset
import numpy as np


class DataSetCatCon(Dataset):
    def __init__(self, data, sequence_length, features_n):
        target_fea = 'ClosePrice'
        # print(data.columns.to_numpy()[::-1].reshape(5,6))
        data = data[data.columns.to_numpy()[::-1]].to_numpy()
        train_y = data[:, -1].copy()
        data[:, -1] = -1

        self.x = data.reshape(-1, sequence_length, features_n)
        self.y = train_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 一个批量就是一天的数据
class DataSetCatCon_2(Dataset):

    def __init__(self, data, sequence_length, features_n, trading_dates=None):
        self.trading_dates = trading_dates
        self.unique_trading_dates = np.sort(np.unique(self.trading_dates))
        data = data[data.columns.to_numpy()[::-1]].to_numpy()
        train_y = data[:, -1].copy()
        data[:, -1] = -1
        self.sequence_length = sequence_length
        self.features_n = features_n
        # self.x = data.reshape(-1, sequence_length, features_n)
        self.x = data.copy()
        self.y = train_y

    def __len__(self):
        return len(self.unique_trading_dates)
        # return 3

    def __getitem__(self, idx):
        # print(idx)
        trading_date = self.unique_trading_dates[int(idx)]
        data_idx = np.where(self.trading_dates == trading_date)
        # date_idx = np.random.permutation(len(np.array(date_idx).squeeze()))[:100]
        x = np.array(self.x[data_idx[0]])
        sort_index = np.argsort(x[:, 162])
        x = x[sort_index]
        x = x.reshape(-1, self.sequence_length, self.features_n)
        y = np.array(self.y[data_idx[0]])
        y = y[sort_index]

        return x, y
