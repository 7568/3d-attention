# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/10/31
Description:
"""
import os
import sys
import logging

import lightgbm.basic
import pandas as pd
from openpyxl.workbook import Workbook
from sklearn.metrics import confusion_matrix, auc, accuracy_score,f1_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
from torch.utils.data import DataLoader

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


def mutil_class_eval_accuracy(y_true, y_test_hat):
    _0_0, _0_1, _0_2, _1_0, _1_1, _1_2, _2_0, _2_1, _2_2 = confusion_matrix(y_true, y_test_hat).ravel()
    print('0：不涨不跌 ， 1：涨，2：跌')
    print('_0_1 ：表示预测为1，但是实际上为0')
    print('_0_0,_0_1,_0_2,_1_0,_1_1,_1_2,_2_0,_2_1,_2_2 : ', _0_0, _0_1, _0_2, _1_0, _1_1, _1_2, _2_0, _2_1, _2_2)

    print(f'test中为0的比例 : {len(y_true[y_true == 0]) / len(y_true)}')
    print(f'test中为1的比例 : {len(y_true[y_true == 1]) / len(y_true)}')
    print(f'test中为2的比例 : {len(y_true[y_true == 2]) / len(y_true)}')

    # error_in_test = mean_squared_error(y_test_hat, np.array(testing_df[target_fea]).reshape(-1, 1))
    print(f'预测为1 且实际为1 ，看涨的准确率: {_1_1 / (_0_1 + _1_1 + _2_1)}')
    # print(f'真实为2中，预测为2 ，看跌的查全率: {_2_2 / (_2_0 + _2_1 + _2_2)}')
    print(f'预测为1，实际为2 ，重大损失的比例: {_2_1 / (_0_1 + _1_1 + _2_1)}')


def reformat_data(training_df, validation_df, testing_df, not_use_pre_data=False):
    """
    训练的时候，前4天的 up_and_down 的值可见，当天的不可见，且设置为-1
    :param training_df:
    :param validation_df:
    :param testing_df:
    :param not_use_pre_data:
    :return:
    """
    target_fea = 'ClosePrice'
    train_x = training_df.copy()
    # train_x = train_x.iloc[:,:-5]
    train_y = training_df[target_fea]

    validation_x = validation_df.copy()
    # validation_x = validation_x.iloc[:,:-5]
    validation_y = validation_df[target_fea]

    testing_x = testing_df.copy()
    # testing_x = testing_x.iloc[:,:-5]
    testing_y = testing_df[target_fea]

    # latest_x = latest_df.copy()
    # latest_x.loc[:, target_fea] = -1
    # latest_y = latest_df[target_fea]
    if not_use_pre_data:
        train_x = train_x.iloc[:, :int(train_x.shape[1] / 5)]
        validation_x = validation_x.iloc[:, :int(validation_x.shape[1] / 5)]
        testing_x = testing_x.iloc[:, :int(testing_x.shape[1] / 5)]
        # latest_x = latest_x.iloc[:, :int(latest_x.shape[1] / 5)]
    train_x.loc[:, target_fea] = 0
    validation_x.loc[:, target_fea] = 0
    testing_x.loc[:, target_fea] = 0
    return train_x, train_y, validation_x, validation_y, testing_x, testing_y

def mse_loss(y_pred, y_val):
    """
    在xgboost中自定义mseloss
    """
    # l(y_val, y_pred) = (y_val-y_pred)**2
    if type(y_val) is lightgbm.basic.Dataset:
        y_val = y_val.get_label()
    grad = 2*(y_val-y_pred)
    hess = np.repeat(2,y_val.shape[0])
    return grad, hess

def show_regression_result(y_test_true, y_test_hat,def_print=True):
    rmse = mean_squared_error(y_test_true, y_test_hat, squared=False)
    mae = mean_absolute_error(y_test_true, y_test_hat)
    if def_print:
        print(f'rmse : {rmse} , mae : {mae}')
    return rmse,mae


def seperate_by_year(df):
    mask_2020 = (df['TradingDate'] < pd.Timestamp('2021-01-01')) & (df['TradingDate'] > pd.Timestamp('2019-12-31'))
    df_year_2020 = df[mask_2020]
    mask_2021 = (df['TradingDate'] < pd.Timestamp('2022-01-01')) & (df['TradingDate'] > pd.Timestamp('2020-12-31'))
    df_year_2021 = df[mask_2021]
    mask_2021 = (df['TradingDate'] < pd.Timestamp('2023-01-01')) & (df['TradingDate'] > pd.Timestamp('2021-12-31'))
    df_year_2022 = df[mask_2021]
    return df_year_2020, df_year_2021, df_year_2022


def analysis_by_year_moneyness(result_df):
    data_2020,data_2021,data_2022 = seperate_by_year(result_df)
    year_infos = {'data_2020':data_2020,'data_2021':data_2021,'data_2022':data_2022}
    for year_str in year_infos:
        # mask_2020 = (df['TradingDate'] < pd.Timestamp('2023-01-01')) & (df['TradingDate'] > pd.Timestamp('2019-12-31'))
        # df_year_2020_2022 = df[mask_2020]
        year_info = year_infos[year_str]
        print(f'{year_str} , the shape is {year_info.shape}')
        moneyness_less_097 = year_info[year_info['moneyness'] <= 0.97]
        # print(f'moneyness_less_097.shape : {moneyness_less_097.shape}')
        moneyness_in_097_103 = year_info.loc[(year_info['moneyness'] > 0.97) & (year_info['moneyness'] <= 1.03)]
        # print(f'moneyness_in_097_103.shape : {moneyness_in_097_103.shape}')
        moneyness_more_103 = year_info[year_info['moneyness'] > 1.03]
        # print(f'moneyness_more_103.shape : {moneyness_more_103.shape}')
        rmse,mae = show_regression_result(moneyness_less_097['y_test_true'], moneyness_less_097['y_test_hat'],def_print=False)
        print(f'moneyness_less_097  rmse : {rmse} , mae : {mae}')
        rmse, mae = show_regression_result(moneyness_in_097_103['y_test_true'], moneyness_in_097_103['y_test_hat'],def_print=False)
        print(f'moneyness_in_097_103  rmse : {rmse} , mae : {mae}')
        rmse, mae = show_regression_result(moneyness_more_103['y_test_true'], moneyness_more_103['y_test_hat'],def_print=False)
        print(f'moneyness_more_103  rmse : {rmse} , mae : {mae}')

def analysis_by_moneyness_maturity(result_df,max_day=210,tc='table'):
    wb = Workbook()
    ws = wb.active
    ws.append(['Moneyness', 'Maturity', 'RMSE','MAE'])
    year_info = result_df
    moneyness_less_097 = year_info[year_info['moneyness'] <= 0.97]
    # print(f'moneyness_less_097.shape : {moneyness_less_097.shape}')
    moneyness_in_097_103 = year_info.loc[(year_info['moneyness'] > 0.97) & (year_info['moneyness'] <= 1.03)]
    # print(f'moneyness_in_097_103.shape : {moneyness_in_097_103.shape}')
    moneyness_more_103 = year_info[year_info['moneyness'] > 1.03]
    # print(f'moneyness_more_103.shape : {moneyness_more_103.shape}')
    moneyness_infos = {'<0.97':moneyness_less_097,'0.97 ~ 1.03':moneyness_in_097_103,'≥1.03':moneyness_more_103,}
    for moneyness_str in moneyness_infos:
        moneyness_info = moneyness_infos[moneyness_str]
        span_days = 30
        for d in range(0,210,span_days):
            maturity = moneyness_info.loc[(moneyness_info['RemainingTerm'] >= d) & (moneyness_info['RemainingTerm'] < (d+span_days))]
            rmse, mae = show_regression_result(maturity['y_test_true'], maturity['y_test_hat'],def_print=False)
            # print(f'{moneyness_str} ,  maturity : {d} ~ {d+span_days}   rmse : {rmse} , mae : {mae}')
            print(rmse)
            ws.append([moneyness_str, f'{d} ~ {d+span_days}', "{:.4g}".format(rmse), "{:.4g}".format(mae)])
        if max_day>270:
            d = 210
            maturity = moneyness_info.loc[
                (moneyness_info['RemainingTerm'] >= d) & (moneyness_info['RemainingTerm'] < (max_day))]
            rmse, mae = show_regression_result(maturity['y_test_true'], maturity['y_test_hat'], def_print=False)
            # print(f'{moneyness_str} ,  maturity : {d} ~ {d+span_days}   rmse : {rmse} , mae : {mae}')
            print(rmse)
            ws.append([moneyness_str, f'{d} ~ {d + max_day}', "{:.4g}".format(rmse), "{:.4g}".format(mae)])
    wb.save(f'{tc}.xlsx')


def load_2_d_data(use_much_features,prepare_home_path,normal_type):

    training_df = pd.read_csv(f'{prepare_home_path}/{normal_type}/training.csv')
    validation_df = pd.read_csv(f'{prepare_home_path}/{normal_type}/validation.csv')
    testing_df = pd.read_csv(f'{prepare_home_path}/{normal_type}/testing.csv', parse_dates=['TradingDate'])
    testing_df_tradingDate = testing_df['TradingDate']
    less_features = ['rate_7_formatted', 'UnderlyingScrtClose', 'ImpliedVolatility',
                     'StrikePrice', 'RemainingTerm', 'ClosePrice']
    cat_features = []
    if use_much_features:
        less_features = ['TheoreticalPrice', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho',
                         'OpenPrice', 'HighPrice', 'LowPrice', 'SettlePrice', 'Change1', 'Change2', 'Volume',
                         'Position', 'Amount', 'PositionChange', 'MainSign',
                         'rate_1_formatted', 'rate_2_formatted', 'rate_3_formatted', 'rate_7_formatted',
                         'rate_14_formatted', 'rate_21_formatted',
                         'AvgPrice', 'ClosePriceChangeRatio', 'SettlePriceChangeRatio', 'Amplitude', 'LimitUp',
                         'LimitDown',
                         'MaintainingMargin', 'ChangeRatio', 'UnderlyingScrtClose', 'ImpliedVolatility',
                         'StrikePrice', 'RemainingTerm', 'ClosePrice']

        cat_features = ['MainSign']
        for i in range(1, 5):
            cat_features.append(f'MainSign_{i}')
    _features = less_features.copy()
    for i in range(1, 5):
        for f in _features:
            less_features.append(f'{f}_{i}')

    training_df = training_df[less_features]
    validation_df = validation_df[less_features]
    testing_df = testing_df[less_features]


    return training_df,validation_df,testing_df,cat_features,testing_df_tradingDate

