# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/6/13
Description:
"""
import argparse
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool,CatBoostRegressor


from library import util


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_file', action='store_true')
    return parser.parse_args()

def train(train_x, train_y, validation_x, validation_y,dataset_name):
    params = {
        'iterations': 5000,
        'depth': 8,
        'learning_rate': 0.015,
        # 'loss_function': '',
        # 'verbose': False,
        'task_type': "GPU",
        'logging_level': 'Verbose',
        'devices': '7',
        'early_stopping_rounds': 20,
        # 'eval_metric':'Accuracy'

    }

    train_pool = Pool(train_x, np.array(train_y).reshape(-1, 1), cat_features=cat_features)
    validation_pool = Pool(validation_x, np.array(validation_y).reshape(-1, 1), cat_features=cat_features)
    test_pool = Pool(testing_x, cat_features=cat_features)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=validation_pool, log_cerr=sys.stderr, log_cout=sys.stdout)
    if opt.log_to_file:
        util.remove_file_if_exists(f'CatBoostRegressor')
        model.save_model("CatBoostRegressor")

        from_file = CatBoostClassifier()

        from_file.load_model("CatBoostRegressor")
    else:
        from_file = model
    model.save_model(f"catBoostRegressor_{dataset_name}")
    # make the prediction using the resulting model
    # y_train_hat = from_file.predict(train_pool)
    # y_validation_hat = from_file.predict(validation_pool)
    # y_test_hat = from_file.predict(test_pool)
    #
    # y_train_true = np.array(train_y).reshape(-1, 1)
    # y_validation_true = np.array(validation_y).reshape(-1, 1)
    # y_test_true = np.array(testing_y).reshape(-1, 1)
    #
    # util.show_regression_result(y_train_true, y_train_hat)
    # util.show_regression_result(y_validation_true, y_validation_hat)
    # util.show_regression_result(y_test_true, y_test_hat)
    # return y_test_true ,y_test_hat

def get_result(train_x, train_y, validation_x, validation_y, testing_x, testing_y,testing_df_tradingDate,max_day=210):
    train(train_x, train_y, validation_x, validation_y, testing_x, testing_y)
    # testing_df.loc[:, 'TradingDate'] = testing_df_tradingDate
    # testing_df.loc[:, 'y_test_true'] = y_test_true
    # testing_df.loc[:, 'y_test_hat'] = y_test_hat
    # spot = testing_df['UnderlyingScrtClose'].to_numpy()
    # strike = testing_df['StrikePrice'].to_numpy()
    # df_year = spot / strike
    # testing_df.loc[:, 'moneyness'] = df_year
    # testing_df.loc[:, 'RemainingTerm'] = testing_df['RemainingTerm'] * 365
    # result_df = testing_df[['TradingDate', 'moneyness', 'RemainingTerm', 'y_test_true', 'y_test_hat']]
    # table_name = 'h_sh_300_moneyness_maturity'
    # util.analysis_by_moneyness_maturity(result_df, max_day, table_name)

# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/ETF50-option/'
PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/index-option/h_sh_300/'
if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('catboost_delta_hedging_v2')
    # NORMAL_TYPE = 'min_max_norm'
    NORMAL_TYPE = 'mean_norm'
    use_much_features = False
    max_depth = 8
    if use_much_features:
        max_depth = 12
    training_df, validation_df, testing_df, cat_features,testing_df_tradingDate = util.load_2_d_data(use_much_features, PREPARE_HOME_PATH,NORMAL_TYPE)
    train_x, train_y, validation_x, validation_y, testing_x, testing_y = util.reformat_data(
        training_df, validation_df, testing_df, not_use_pre_data=False)
    train(train_x, train_y, validation_x, validation_y,'h_sh_300')
    # get_result(train_x, train_y, validation_x, validation_y, testing_x, testing_y,testing_df_tradingDate,max_day=360)
    PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/ETF50-option/'
    training_df, validation_df, testing_df, cat_features, testing_df_tradingDate = util.load_2_d_data(use_much_features,
                                                                                                      PREPARE_HOME_PATH,
                                                                                                      NORMAL_TYPE)
    train_x, train_y, validation_x, validation_y, testing_x, testing_y = util.reformat_data(
        training_df, validation_df, testing_df, not_use_pre_data=False)
    train(train_x, train_y, validation_x, validation_y,'ETF50')
    # get_result(train_x, train_y, validation_x, validation_y, testing_x, testing_y,testing_df_tradingDate,max_day=210)



"""
ETF50
rmse : 0.05204370022769998 , mae : 0.03568510798598349
rmse : 0.06466544219738538 , mae : 0.04434767865975838
rmse : 0.06364029258890669 , mae : 0.04269551978408703
"""

"""
h_sh_300
rmse : 0.060300571994874305 , mae : 0.0375644887299065
rmse : 0.06791130831134849 , mae : 0.04043631169457862
rmse : 0.06515894501661627 , mae : 0.039700963170050695
"""