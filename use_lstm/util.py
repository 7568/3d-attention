# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/10/31
Description:
"""
import os
import sys
import logging


import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, mean_squared_error, mean_absolute_error

import numpy as np
from torch.utils.data import DataLoader

from data_openml import DataSetCatCon
from data_openml import DataSetCatCon_2


def remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def touch(fname, times=None):
    fhandle = open(fname, 'a')
    try:
        os.utime(fname, times)
    finally:
        fhandle.close()


def init_log(file_name='tmp'):
    if not os.path.exists(f'log/'):
        os.mkdir(f'log/')
    if not os.path.exists(f'log/{file_name}_std_out.log'):
        touch(f'log/{file_name}_std_out.log')
    if not os.path.exists(f'log/{file_name}_debug_info.log'):
        touch(f'log/{file_name}_debug_info.log')
    sys.stderr = open(f'log/{file_name}_std_out.log', 'a')
    sys.stdout = open(f'log/{file_name}_std_out.log', 'a')
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(f'log/{file_name}_debug_info.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def binary_eval_accuracy(y_true, y_test_hat):
    tn, fp, fn, tp = confusion_matrix(y_true, y_test_hat).ravel()
    print('0：不涨 ， 1：涨')
    print('tn, fp, fn, tp', tn, fp, fn, tp)

    print(f'test中为1的比例 : {y_true.sum() / len(y_true)}')
    print(f'test中为0的比例 : {(1 - y_true).sum() / len(y_true)}')

    # error_in_test = mean_squared_error(y_test_hat, np.array(testing_df[target_fea]).reshape(-1, 1))
    print(f'查准率 - 预测为1 且实际为1 ，看涨的准确率: {tp / (tp + fp)}')
    print(f'查全率 - 实际为1，预测为1 : {tp / (tp + fn)}')
    f_1 = f1_score(y_true, y_test_hat, average="binary")
    print(f'F1 = {f_1}')
    # print(f'AUC：{auc(y_true,y_test_hat)}')
    print(f'总体准确率：{accuracy_score(y_true, y_test_hat)}')


def load_sequence_data(use_much_features,prepare_home_path,normal_type,opt):

    training_df = pd.read_csv(f'{prepare_home_path}/{normal_type}/training.csv')
    validation_df = pd.read_csv(f'{prepare_home_path}/{normal_type}/validation.csv')
    testing_df = pd.read_csv(f'{prepare_home_path}/{normal_type}/testing.csv')
    training_trading_dates = training_df['TradingDate'].copy()
    validation_trading_dates = validation_df['TradingDate'].copy()
    testing_trading_dates = testing_df['TradingDate'].copy()
    less_features = ['ClosePrice', 'rate_7_formatted', 'UnderlyingScrtClose', 'ImpliedVolatility', 'StrikePrice',
                     'RemainingTerm']
    if use_much_features:
        less_features = ['ClosePrice', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho',
                         'OpenPrice', 'HighPrice', 'LowPrice', 'SettlePrice', 'Change1', 'Change2', 'Volume',
                         'Position', 'Amount', 'PositionChange', 'MainSign',
                         'rate_1_formatted', 'rate_2_formatted', 'rate_3_formatted', 'rate_7_formatted',
                         'rate_14_formatted', 'rate_21_formatted',
                         'AvgPrice', 'ClosePriceChangeRatio', 'SettlePriceChangeRatio', 'Amplitude', 'LimitUp',
                         'LimitDown',
                         'MaintainingMargin', 'ChangeRatio', 'UnderlyingScrtClose', 'ImpliedVolatility',
                         'StrikePrice', 'RemainingTerm']
    _features = less_features.copy()
    for i in range(1, 5):
        for f in _features:
            less_features.append(f'{f}_{i}')
    training_df = training_df[less_features]
    validation_df = validation_df[less_features]
    testing_df = testing_df[less_features]
    sequence_length = 5
    features_n = len(less_features)//sequence_length
    if opt.batchsize==1:
        train_ds = DataSetCatCon_2(training_df,sequence_length,features_n,training_trading_dates)
        validation_ds = DataSetCatCon_2(validation_df,sequence_length,features_n,validation_trading_dates)
        testing_ds = DataSetCatCon_2(testing_df,sequence_length,features_n,testing_trading_dates)
    else:
        train_ds = DataSetCatCon(training_df, sequence_length, features_n)
        validation_ds = DataSetCatCon(validation_df, sequence_length, features_n)
        testing_ds = DataSetCatCon(testing_df, sequence_length, features_n)

    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)
    validationloader = DataLoader(validation_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)
    testingloader = DataLoader(testing_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)
    return trainloader, validationloader, testingloader, features_n

def show_regression_result(y_test_true, y_test_hat):
    rmse = mean_squared_error(y_test_true, y_test_hat, squared=False)
    mae = mean_absolute_error(y_test_true, y_test_hat)
    print(f'rmse : {rmse} , mae : {mae}')
    return rmse,mae