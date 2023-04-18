import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error

import util


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_file', action='store_true')
    return parser.parse_args()


PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/ETF50-option/'
if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('B_S_price')
    # NORMAL_TYPE = 'min_max_norm'
    NORMAL_TYPE = 'mean_norm'
    testing_df = pd.read_csv(f'{PREPARE_HOME_PATH}/{NORMAL_TYPE}/testing.csv')

    less_features=['TheoreticalPrice','ClosePrice']
    testing_df = testing_df[less_features]


    y_test_hat = testing_df['TheoreticalPrice'].to_numpy()

    y_test_true = testing_df['ClosePrice'].to_numpy()

    util.show_regression_result(y_test_true,y_test_hat)
# rmse : 0.05977277613982846 , mae : 0.04382507086604558