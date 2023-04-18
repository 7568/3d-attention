# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import argparse

import lightgbm as lgb
import pandas as pd
from lightgbm import early_stopping, log_evaluation
import numpy as np
import util


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_file', action='store_true')
    opt = parser.parse_args()
    return opt


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/ETF50-option/'

if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('lgboost_delta_hedging_v2')
    # NORMAL_TYPE = 'min_max_norm'
    # NORMAL_TYPE = 'no_norm'
    NORMAL_TYPE = 'mean_norm'
    training_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/training.csv')
    validation_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/validation.csv')
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')
    less_features = ['RisklessRate', 'UnderlyingScrtClose', 'ImpliedVolatility',
                     'StrikePrice', 'RemainingTerm','ClosePrice']

    cat_features = []
    for i in range(1, 5):
        less_features.append(f'RisklessRate_{i}')
        less_features.append(f'UnderlyingScrtClose_{i}')
        less_features.append(f'ImpliedVolatility_{i}')
        less_features.append(f'StrikePrice_{i}')
        less_features.append(f'RemainingTerm_{i}')
        # less_features.append(f'TheoreticalPrice_{i}')
        less_features.append(f'ClosePrice_{i}')
    training_df = training_df[less_features]
    validation_df = validation_df[less_features]
    testing_df = testing_df[less_features]
    train_x, train_y, validation_x, validation_y, testing_x, testing_y = util.reformat_data(
        training_df, validation_df, testing_df, not_use_pre_data=False)

    params = {'objective': 'regression',
              # 'boosting': 'gbdt',
              'learning_rate': 0.01,
              'max_depth': 8,
              # 'num_leaves': 2 ** 8,
              'lambda_l1': 0.5,
              'lambda_l2': 0.5,
              'feature_fraction': 0.75,
              'bagging_fraction': 0.75,
              'bagging_freq': 20,
              # 'force_col_wise': True,
              # 'metric': 'binary_logloss',
              # 'num_classes': 3
              }

    num_round = 5000
    early_s_n = 20
    train_data = lgb.Dataset(train_x, train_y)
    validation_data = lgb.Dataset(validation_x, validation_y)
    bst = lgb.train(params, train_data, num_round,valid_sets=[validation_data],
                    callbacks=[early_stopping(early_s_n), log_evaluation()])
    if opt.log_to_file:

        util.remove_file_if_exists(f'lgboostClassifier')
        bst.save_model('lgboostClassifier', num_iteration=bst.best_iteration)
        bst_from_file = lgb.Booster(model_file='lgboostClassifier')
    else:
        bst_from_file = bst
    y_train_hat = bst_from_file.predict(train_x, num_iteration=bst.best_iteration)
    y_validation_hat = bst_from_file.predict(validation_x, num_iteration=bst.best_iteration)
    y_test_hat = bst_from_file.predict(testing_x, num_iteration=bst.best_iteration)

    util.show_regression_result(train_y, y_train_hat)
    util.show_regression_result(validation_y, y_validation_hat)
    util.show_regression_result(testing_y, y_test_hat)

# rmse : 0.04654823075221435 , mae : 0.03417544898479541