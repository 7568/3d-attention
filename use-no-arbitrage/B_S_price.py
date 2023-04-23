import argparse

import pandas as pd

from library import util


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_file', action='store_true')
    return parser.parse_args()

def create_table(table_name,max_day=210):
    NORMAL_TYPE = 'mean_norm'
    testing_df = pd.read_csv(f'{DATA_HOME_PATH}/{NORMAL_TYPE}/testing.csv', parse_dates=['TradingDate'])

    y_test_hat = testing_df['TheoreticalPrice'].to_numpy()
    y_test_true = testing_df['ClosePrice'].to_numpy()
    testing_df.loc[:, 'y_test_true'] = y_test_true
    testing_df.loc[:, 'y_test_hat'] = y_test_hat
    spot = testing_df['UnderlyingScrtClose'].to_numpy()
    strike = testing_df['StrikePrice'].to_numpy()
    df_year = spot / strike
    testing_df.loc[:, 'moneyness'] = df_year
    testing_df.loc[:, 'RemainingTerm'] = testing_df['RemainingTerm'] * 365
    result_df = testing_df[['TradingDate', 'moneyness', 'RemainingTerm', 'y_test_true', 'y_test_hat']]
    util.analysis_by_moneyness_maturity(result_df,max_day, table_name)

# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/ETF50-option/'
# PREPARE_HOME_PATH = '/home/liyu/data/hedging-option/20170101-20230101/index-option/h_sh_300/'
HOME_PATH = f'/home/liyu/data/hedging-option/20170101-20230101/'
if __name__ == '__main__':
    opt = init_parser()
    if opt.log_to_file:
        logger = util.init_log('B_S_price')
    # NORMAL_TYPE = 'min_max_norm'


    OPTION_SYMBOL = 'index-option/h_sh_300'
    DATA_HOME_PATH = HOME_PATH + "/" + OPTION_SYMBOL + "/"
    # reformatt_data()

    create_table('h_sh_300_moneyness_maturity',max_day=360)

    OPTION_SYMBOL = 'ETF50-option'
    DATA_HOME_PATH = HOME_PATH + "/" + OPTION_SYMBOL + "/"
    # reformatt_data()
    create_table('ETF50_moneyness_maturity')

