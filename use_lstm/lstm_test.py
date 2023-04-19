# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/10/28
Description:
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_openml import DataSetCatCon
from use_lstm import util

torch.manual_seed(100)


def train_test(df, test_periods):
    train = df[:-test_periods].values
    test = df[-test_periods:].values
    return train, test


class LSTM(nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        self.linear = nn.Sequential(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # print(hn.shape)

        predictions = self.linear(lstm_out[:, -1, :])

        return predictions


def get_scheduler(epochs, optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[20, 30, 50, 80], gamma=0.1)
    return scheduler


def get_test_result(model, validationloader, device):
    model.eval()
    with torch.no_grad():
        y_hats = []
        y_s = []
        for x, y in tqdm(validationloader):
            x = x.to(device).float()
            y = y.to(device).float()
            y_hat = model(x)
            y_hats = np.append(y_hats, y_hat.squeeze().detach().cpu().numpy())
            y_s = np.append(y_s, y.detach().cpu().numpy())
        rmse, mae = util.show_regression_result(y_s, y_hats)
        return rmse, mae


def train_model(use_much_features):
    device = torch.device(f"cuda:6")
    # device = torch.device(f"cpu")
    NORMAL_TYPE = 'mean_norm'
    print(f"Device is {device}.")
    trainloader, validationloader, testingloader, feature_num = util.load_sequence_data(use_much_features,
                                                                                        PREPARE_HOME_PATH, NORMAL_TYPE,
                                                                                        opt)
    test_periods = 1
    model = LSTM(input_size=feature_num, hidden_size=opt.hidden_size, output_size=test_periods,
                 num_layers=opt.num_layers).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = get_scheduler(opt.epochs, optimizer)
    loss_list = []
    lr_list = []
    min_validation_loss = 99999
    no_change_times = 0
    for epoch in range(opt.epochs + 1):
        model.train()
        one_epoch_loss = []
        for x, y in tqdm(trainloader):
            x = x.to(device).float()
            y = y.to(device).float()
            y_hat = model(x)
            optimizer.zero_grad()
            loss = criterion(y_hat.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            one_epoch_loss.append(loss.item())
        loss_list.append(np.array(one_epoch_loss).mean())
        # scheduler.step(30)
        lr_list.append(scheduler.get_last_lr())
        # for g in optimizer.param_groups:
        #     g['lr'] = 0.001
        scheduler.step()
        if epoch % 1 == 0:
            print(f'epoch: {epoch:4} loss:{loss.item():10.9f} , lr={optimizer.param_groups[0]["lr"]}')

        rmse, mae = get_test_result(model, validationloader, device)

        print(f'validation rmse : {rmse} , mae : {mae}')
        no_change_times += 1
        if rmse < min_validation_loss:
            min_validation_loss = rmse
            no_change_times = 0
        if no_change_times > 19:
            break
        # predictions = model(train_scaled.to(device), None, device)
    rmse, mae = get_test_result(model, testingloader, device)
    print(f'testing rmse : {rmse} , mae : {mae}')


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_file', action='store_true')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batchsize', default=128, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)  # The number of features in the hidden state h
    parser.add_argument('--num_layers', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    return parser.parse_args()


# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/ETF50-option/'
PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/index-option/h_sh_300/'
if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('lstm')
    use_much_features = True
    train_model(use_much_features)

"""
h_sh_300
validation rmse : 0.04811541363048127 , mae : 0.0251478114582285
testing rmse : 0.04673098007204823 , mae : 0.02567159558187799
"""

"""
ETF50
validation rmse : 0.026531293950002928 , mae : 0.017034916364539125
testing rmse : 0.026283616057762242 , mae : 0.01760636860360867
"""
