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

def predict(testing_x, testing_y,dataset_name):

    test_pool = Pool(testing_x, cat_features=cat_features)

    model = CatBoostRegressor()
    model.load_model(f"catBoostRegressor_{dataset_name}")
    from_file = model
    # make the prediction using the resulting model

    y_test_hat = from_file.predict(test_pool)


    y_test_true = np.array(testing_y).reshape(-1, 1)


    util.show_regression_result(y_test_true, y_test_hat)
    return y_test_true ,y_test_hat

def get_result( testing_x, testing_y,dataset_name,testing_df_tradingDate,max_day=210):
    y_test_true, y_test_hat = predict( testing_x, testing_y,dataset_name)
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
    util.analysis_by_moneyness_maturity(result_df, max_day, table_name)

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
    _, __, ___, ____, testing_x, testing_y = util.reformat_data(
        training_df, validation_df, testing_df, not_use_pre_data=False)
    dataset_name='h_sh_300'
    get_result(testing_x, testing_y,dataset_name,testing_df_tradingDate,max_day=360)
    PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/ETF50-option/'
    training_df, validation_df, testing_df, cat_features, testing_df_tradingDate = util.load_2_d_data(use_much_features,
                                                                                                      PREPARE_HOME_PATH,
                                                                                                      NORMAL_TYPE)
    _, __, ___, ____, testing_x, testing_y = util.reformat_data(
        training_df, validation_df, testing_df, not_use_pre_data=False)
    dataset_name = 'ETF50'
    get_result(testing_x, testing_y,dataset_name,testing_df_tradingDate,max_day=210)



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