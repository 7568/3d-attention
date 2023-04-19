# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import argparse

import lightgbm as lgb
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from library import util


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_file', action='store_true')
    opt = parser.parse_args()
    return opt


# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/ETF50-option/'
PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/index-option/h_sh_300/'

if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('lgboost_delta_hedging_v2')
    # NORMAL_TYPE = 'min_max_norm'
    # NORMAL_TYPE = 'no_norm'
    NORMAL_TYPE = 'mean_norm'
    use_much_features=True
    max_depth = 8
    if use_much_features:
        max_depth = 12
    training_df, validation_df, testing_df,cat_features = util.load_2_d_data(use_much_features,PREPARE_HOME_PATH,NORMAL_TYPE)
    train_x, train_y, validation_x, validation_y, testing_x, testing_y = util.reformat_data(
        training_df, validation_df, testing_df, not_use_pre_data=False)

    params = {'objective': 'regression',
              # 'boosting': 'gbdt',
              'learning_rate': 0.01,
              'max_depth': max_depth,
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
    bst = lgb.train(params, train_data, num_round,valid_sets=[validation_data],categorical_feature=cat_features,
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

# rmse : 0.06690369309873591 , mae : 0.044401452548709676

"""
ETF50
rmse : 0.02393914790521441 , mae : 0.015409243299522535
rmse : 0.04173552429266133 , mae : 0.025952743145917594
rmse : 0.04322508458278481 , mae : 0.026519479395672178
"""

"""
h_sh_300
rmse : 0.03830374486264224 , mae : 0.022777631513925988
rmse : 0.05359733846529638 , mae : 0.029862857466974764
rmse : 0.049830594751371644 , mae : 0.029150886975433235
"""