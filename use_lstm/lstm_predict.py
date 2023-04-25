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

from library import util as lib_util
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


def predict(use_much_features,dataset_name,max_day):
    device = torch.device(f"cuda:6")
    # device = torch.device(f"cpu")
    NORMAL_TYPE = 'mean_norm'
    print(f"Device is {device}.")
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv', parse_dates=['TradingDate'])
    testing_df_tradingDate = testing_df['TradingDate']
    _, __, testingloader, feature_num = util.load_sequence_data(use_much_features,
                                                                                        PREPARE_HOME_PATH, NORMAL_TYPE,
                                                                                        opt)
    test_periods = 1
    model = LSTM(input_size=feature_num, hidden_size=opt.hidden_size, output_size=test_periods,
                 num_layers=opt.num_layers).to(device)
    checkpoint = torch.load( f'lstm_best_model_{dataset_name}')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        y_hats = []
        y_s = []
        for x, y in tqdm(testingloader):
            x = x.to(device).float()
            y = y.to(device).float()
            y_hat = model(x)
            y_hats = np.append(y_hats, y_hat.squeeze().detach().cpu().numpy())
            y_s = np.append(y_s, y.detach().cpu().numpy())
    get_result(testing_df,np.array(y_s),np.array(y_hats),dataset_name,testing_df_tradingDate,max_day)


def get_result( testing_df,y_test_true, y_test_hat,dataset_name,testing_df_tradingDate,max_day=210):
    testing_df.loc[:, 'TradingDate'] = testing_df_tradingDate
    testing_df.loc[:, 'y_test_true'] = y_test_true
    testing_df.loc[:, 'y_test_hat'] = y_test_hat
    spot = testing_df['UnderlyingScrtClose'].to_numpy()
    strike = testing_df['StrikePrice'].to_numpy()
    df_year = spot / strike
    testing_df.loc[:, 'moneyness'] = df_year
    testing_df.loc[:, 'RemainingTerm'] = np.round((testing_df['RemainingTerm'] * 365).to_numpy())
    result_df = testing_df[['TradingDate', 'moneyness', 'RemainingTerm', 'y_test_true', 'y_test_hat']]
    table_name = f'{dataset_name}_moneyness_maturity'
    lib_util.analysis_by_moneyness_maturity(result_df, max_day, table_name)

def load_model(use_much_features,dataset_name):
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


    return model



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
PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/index-option/h_sh_300_option/'
if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('lstm')
    use_much_features = False
    predict(use_much_features,'h_sh_300',360)
    PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/ETF50-option/'
    predict(use_much_features,'ETF50',210)

"""
h_sh_300
use_much_features = True
validation rmse : 0.004036936887173826 , mae : 0.002130362190093025
testing rmse : 0.004675756610792724 , mae : 0.002104783016949187

use_much_features = False
validation rmse : 0.03754063364282469 , mae : 0.017040287596522707
testing rmse : 0.0429548307882293 , mae : 0.017727977690909127
"""

"""
ETF50
use_much_features = True
validation rmse : 0.009622219439846227 , mae : 0.0065314762810466465
testing rmse : 0.009738441974197045 , mae : 0.006426467528721215

use_much_features = False
validation rmse : 0.026531293950002928 , mae : 0.017034916364539125
testing rmse : 0.026283616057762242 , mae : 0.01760636860360867
"""
