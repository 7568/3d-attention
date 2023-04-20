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


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/ETF50-option/'
# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/index-option/h_sh_300/'

if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('lgboost_delta_hedging_v2')
    # NORMAL_TYPE = 'min_max_norm'
    # NORMAL_TYPE = 'no_norm'
    NORMAL_TYPE = 'mean_norm'
    use_much_features=False
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

use_much_features=False
rmse : 0.027641644780524926 , mae : 0.01796221438979232
rmse : 0.04743250130208135 , mae : 0.02912669047821163
rmse : 0.04534953518031237 , mae : 0.028514477419322594

use_much_features=True
rmse : 0.01736346811536749 , mae : 0.008533473793200878
rmse : 0.01877766552244604 , mae : 0.009373629487655265
rmse : 0.017370084145710004 , mae : 0.008981620007181564
"""

"""
h_sh_300

use_much_features=False
rmse : 0.04006593908711819 , mae : 0.02396901578588577
rmse : 0.050793993961330774 , mae : 0.02995080132211512
rmse : 0.05587283074117626 , mae : 0.03170890680447272

use_much_features=True
rmse : 0.0108516674012833 , mae : 0.006366965456758745
rmse : 0.019985259823927952 , mae : 0.008049306607088897
rmse : 0.01894083938037142 , mae : 0.008088809522611892
"""