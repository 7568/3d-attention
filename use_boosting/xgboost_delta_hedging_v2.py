# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import argparse
import copy
import math
import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import util
from sklearn.metrics import mean_squared_error,f1_score,confusion_matrix,accuracy_score

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_file', action='store_true')
    opt = parser.parse_args()
    return opt


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/ETF50-option/'
if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('xgboost_delta_hedging_v2')
    NORMAL_TYPE = 'mean_norm'
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv')
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv')
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')
    less_features = ['rate_7_formatted', 'UnderlyingScrtClose', 'HistoricalVolatility', 'ImpliedVolatility',
                     'StrikePrice', 'RemainingTerm', 'TheoreticalPrice', 'ClosePrice']

    cat_features = []
    for i in range(1, 5):
        less_features.append(f'rate_7_formatted_{i}')
        less_features.append(f'UnderlyingScrtClose_{i}')
        less_features.append(f'HistoricalVolatility_{i}')
        less_features.append(f'StrikePrice_{i}')
        less_features.append(f'RemainingTerm_{i}')
        less_features.append(f'ClosePrice_{i}')
    training_df = training_df[less_features]
    validation_df = validation_df[less_features]
    testing_df = testing_df[less_features]
    train_x, train_y, validation_x, validation_y, testing_x, testing_y= util.reformat_data(
        training_df,validation_df, testing_df, not_use_pre_data=False)

    params = {
        # 'objective': 'binary:logistic',
        # 'objective': 'reg:squarederror',
        # 'objective': util.mse_loss,
        # 'objective': mae_loss,
        # 'objective': pseudo_huber_loss,
        'n_estimators': 5000,
        'max_depth': 5,
        'learning_rate': 0.01,
        'tree_method': 'hist',
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'use_label_encoder': False,
        # 'disable_default_eval_metric': 1
        # 'eval_metric': '1-f1'

    }

    model = xgb.XGBRegressor(**params)

    model.fit(train_x.to_numpy(), train_y,
              eval_set=[(validation_x.to_numpy(), np.array(validation_y))],
              early_stopping_rounds=20)
    if opt.log_to_file:

        util.remove_file_if_exists(f'XGBRegressor')
        model.save_model('XGBRegressor')
        model_from_file = xgb.XGBRegressor()
        model_from_file.load_model('XGBRegressor')
    else:
        model_from_file = model
    # Predict on x_test
    y_test_hat = model_from_file.predict(np.ascontiguousarray(testing_x.to_numpy()))

    util.show_regression_result(np.array(testing_y),y_test_hat)
# rmse : 0.04264860653382347 , mae : 0.032187609774597406