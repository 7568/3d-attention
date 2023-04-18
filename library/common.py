import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# This file contains functions that calculate and inspect the hedging error.

def print_removal(before_size, cur_size, ori_size, issue):
    print(
        f'{issue}. {before_size - cur_size} samples ({(before_size - cur_size) / before_size * 100:.2f}%) are removed. We have {cur_size / ori_size * 100:.2f}% of original data left, yielding a size of {cur_size}.')


def calc_pnl(
        df, delta,
        V1='V1_n'
):
    """
    This method calculates PnL. We assume short option.
    :param V1: Target to compare to.
    :return: Series pnl
    """
    s0, s1 = df['S0_n'], df['S1_n']
    v0, v1 = df['V0_n'], df[V1]
    on_return = df['on_ret']

    v1_hat = (v0 - delta * s0) * on_return + delta * s1
    return (v1_hat - v1)


def store_pnl(
        df, delta,
        pnl_path,
        V1='V1_n'
):
    delta = delta[df['Is_In_Some_Test']]
    df = df[df['Is_In_Some_Test']]

    # Cap or floor PNL by security type
    bl_c = df['cp_int'] == 0
    delta[bl_c] = np.maximum(delta[bl_c], 0.)
    delta[~bl_c] = np.minimum(delta[~bl_c], 0.)

    cols = [x for x in df.columns if x in ['ExecuteTime0', 'Aggressorside']]
    cols += ['cp_int', 'date']
    df_res = df[cols].copy()
    df_res['delta'] = delta
    df_res['PNL'] = calc_pnl(df, delta, V1=V1)
    df_res['M0'] = df['M0'].copy()
    df_res['tau0'] = df['tau0'].copy()

    df_res['testperiod'] = np.nan
    # In addition, we want to record which test period the pnl is from.
    max_period = max([int(s[6:]) for s in df.columns if 'period' in s])
    for i in range(0, max_period + 1):
        bl = df['period{}'.format(i)] == 2
        df_res.loc[bl, 'testperiod'] = i

    df_res.to_csv(pnl_path)


def calc_pnl_two_assets(df, delta, eta, V1='V1_n'):
    """ 
    This methods calcuate the PnL, given a strategy of two 
    hedging instruments.
    For the moment, we use underlying and the ATM one-month option.
    """
    s0, s1 = df['S0_n'], df['S1_n']
    v0, v1 = df['V0_n'], df[V1]
    atm0, atm1 = df['V0_atm_n'], df['V1_atm_n']
    on_return = df['on_ret']

    v1_hat = (v0 - delta * s0 - eta * atm0) * on_return + delta * s1 + eta * atm1
    return v1_hat - v1


def store_pnl_two_assets(df, delta, eta, pnl_path, V1='V1_n'):
    delta = delta[df['Is_In_Some_Test']]
    eta = eta[df['Is_In_Some_Test']]
    df = df[df['Is_In_Some_Test']]

    cols = [x for x in df.columns if x in ['ExecuteTime0', 'Aggressorside']]
    cols += ['cp_int', 'date']
    df_res = df[cols].copy()
    df_res['delta'] = delta
    df_res['eta'] = eta
    df_res['PNL'] = calc_pnl_two_assets(df, delta, eta, V1=V1)
    df_res['M0'] = df['M0'].copy()
    df_res['tau0'] = df['tau0'].copy()

    df_res['testperiod'] = np.nan
    # In addition, we want to record which test period the pnl is from.
    max_period = max([int(s[6:]) for s in df.columns if 'period' in s])
    for i in range(0, max_period + 1):
        bl = df['period{}'.format(i)] == 2
        df_res.loc[bl, 'testperiod'] = i

    df_res.to_csv(pnl_path)


class Inspector:
    def __init__(self):
        pass

    def loadPnl(self, path, measure, op_type=None):
        df = pd.read_csv(path, index_col=0)

        bl = self.choose_op_type(df, op_type)

        if measure == 'mse':
            return (df.loc[bl, 'PNL'] ** 2).mean()
        elif measure == 'mean':
            return (df.loc[bl, 'PNL']).mean()
        elif measure == 'median':
            return (df.loc[bl, 'PNL']).median()
        elif measure == 'lower5%VaR':
            return (df.loc[bl, 'PNL']).quantile(0.05)
        elif measure == 'upper95%VaR':
            return (df.loc[bl, 'PNL']).quantile(0.95)
        else:
            raise NotImplementedError('The given measure is not implemented!')

    def choose_op_type(self, df, op_type):
        if op_type == 'call':
            bl = df['cp_int'] == 0
        elif op_type == 'put':
            bl = df['cp_int'] == 1
        else:
            bl = df['cp_int'].notna()
        return bl

    def evalPnls(self, df_dirs, aggregating, measure, op_type=None):
        """
        Params:
        =========================
        aggregating: the aggregating metod over all PNL files
        measure: the measure to evaluate on each PNL file.
        """
        rows, cols = df_dirs.index, df_dirs.columns
        sub_cols = ['Absolute', '%Change']
        cols_indices = pd.MultiIndex.from_product([cols, sub_cols], names=['setup', 'value'])
        df_res = pd.DataFrame(index=rows, columns=cols_indices)

        for r, c in list(itertools.product(rows, cols)):
            directory = os.fsencode(df_dirs.loc[r, c] + 'pnl/')
            res = []
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    filename = os.fsdecode(directory + file)
                    if filename.endswith(".csv"):
                        res.append(self.loadPnl(filename, measure, op_type))

                if aggregating == 'mean':
                    df_res.loc[r, (c, 'Absolute')] = sum(res) / len(res)
                else:
                    raise NotImplementedError('The given aggregating is not implemented!')
            else:
                df_res.loc[r, c] = np.nan

        bs_name = [x for x in df_dirs.index.tolist() if 'BS_Benchmark' in x][0]
        for c in cols:
            tmp = (df_res.loc[:, (c, 'Absolute')] - df_res.loc[bs_name, (c, 'Absolute')]) / \
                  df_res.loc[bs_name, (c, 'Absolute')] * 100.
            tmp = tmp.astype(np.float)
            df_res.loc[:, (c, '%Change')] = tmp.round(2)
            df_res.loc[:, (c, 'Absolute')] = (100 * df_res.loc[:, (c, 'Absolute')]).astype(np.float).round(3)  # JR

        return df_res

    def eval_single_exp(self, dirs_dict, measure, op_type=None):
        """
        load each PNL file in the `directory` and return a list of measurements.
        """
        df_res = pd.DataFrame()
        for y, x in dirs_dict.items():
            directory = x + '/pnl/'
            res = []
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    filename = os.fsdecode(directory + file)
                    if filename.endswith(".csv"):
                        res.append(self.loadPnl(filename, measure, op_type))

                df_res[y] = res
            else:
                df_res[y] = np.nan

        return df_res


class PnlLoader:
    def __init__(self, dirs_dict=None):
        self.pnl = None
        self.record = pd.DataFrame()
        self.dirs_dict = dirs_dict

    def load_real_pnl(self, idx=None):
        """
        Given a dictionary of paths, return a dictionary of pnl files, 
        with the same keys.
        """
        res = {}
        for name, x in self.dirs_dict.items():
            if idx is None:
                res[name] = f'{x}pnl/pnl.csv'
            else:
                res[name] = f'{x}pnl/pnl{idx}.csv'
        self.pnl = {}
        for key, path in res.items():
            self.pnl[key] = pd.read_csv(path, index_col=0)

    def load_aggregate_simulation(self, num_test):
        self.pnl = {}
        for name, x in self.dirs_dict.items():
            directory = f'{x}pnl/'
            df = pd.DataFrame()
            for i in range(num_test):
                filename = directory + f'pnl{i}.csv'
                df_add = pd.read_csv(filename, index_col=0)
                df_add.drop(columns=['testperiod'], inplace=True)
                # for simulation data, we use `testperiod` to index test sets.
                df_add['testperiod'] = i
                df = df.append(df_add)
            df = df.reset_index()
            self.pnl[name] = df


# used mostly for plots. summarizes MSHEs for different models
class LocalInspector(PnlLoader):

    def plug_existing(self, pnl):
        self.pnl = pnl

    def choose_op_type(self, df, op_type):
        if op_type == 'call':
            bl = df['cp_int'] == 0
        elif op_type == 'put':
            bl = df['cp_int'] == 1
        else:
            bl = df['cp_int'].notna()
        return bl

    def compare_period(self, op_type=None):
        """
        In this method, pnl is aggregated for each period.
        """
        for key, pnl in self.pnl.items():
            max_period = int(max(pnl['testperiod']))
            for i in range(max_period + 1):
                bl = pnl['testperiod'] == i
                bl_ = self.choose_op_type(pnl, op_type)
                bl = bl & bl_
                self.record.loc[i, 'num_samples'] = bl.sum()
                self.record.loc[i, key] = (pnl.loc[bl, 'PNL'] ** 2).mean()

        return self.record


def compare_pair(daily_mshe, first, second, trunc_qs):
    N = daily_mshe.shape[0]
    print('Size of N:', N)
    diff = daily_mshe[first] - daily_mshe[second]
    for q in trunc_qs:
        cap = diff.abs().quantile(q)

        truncated_diff = np.maximum(np.minimum(diff, cap), -cap)
        zscore = truncated_diff.mean() / truncated_diff.std() * np.sqrt(N)
        print(f'Mean difference is {truncated_diff.mean()}')
        print(f'Std is {truncated_diff.std()}')
        print(f'Z-score after truncating at {q} is {zscore}')

        truncated_diff.plot()
        plt.show()
        truncated_diff.plot(kind='hist', logy=True, bins=100)
        plt.show()


def truncate_daily_mshe(daily_mshe, first, second, q):
    diff = daily_mshe[first] - daily_mshe[second]
    cap = diff.abs().quantile(q)
    truncated_diff = np.maximum(np.minimum(diff, cap), -cap)
    return truncated_diff


def get_zscore(daily_mshe, first, second, q):
    N = daily_mshe.shape[0]
    truncated_diff = truncate_daily_mshe(daily_mshe, first, second, q)
    zscore = truncated_diff.mean() / truncated_diff.std() * np.sqrt(N)
    return zscore


def get_z_confidence(daily_mshe, first, second, q):
    N = daily_mshe.shape[0]
    truncated_diff = truncate_daily_mshe(daily_mshe, first, second, q)
    up = truncated_diff.mean() + 2 * truncated_diff.std() / np.sqrt(N)
    down = truncated_diff.mean() - 2 * truncated_diff.std() / np.sqrt(N)
    return down, up


def chunks(data, cpu_num):
    x = int(len(data) / cpu_num)
    count_x = cpu_num
    y = 0
    if int(len(data) / cpu_num) != len(data) / cpu_num:
        y = x + 1
        count_y = len(data) - x * cpu_num
        count_x = cpu_num - count_y

    _chunk = []
    for i in range(0, x * count_x, x):
        _end = i + x
        if _end > len(data):
            _end = len(data)
        _chunk.append(data.iloc[i:_end])
    if int(len(data) / cpu_num) != len(data) / cpu_num:
        for i in range(x * count_x, len(data), y):
            _end = i + y
            if _end > len(data):
                _end = len(data)
            _chunk.append(data.iloc[i:_end])
    return _chunk


def chunks_np(data, cpu_num):
    x = int(len(data) / cpu_num)
    count_x = cpu_num
    y = 0
    if int(len(data) / cpu_num) != len(data) / cpu_num:
        y = x + 1
        count_y = len(data) - x * cpu_num
        count_x = cpu_num - count_y

    _chunk = []
    for i in range(0, x * count_x, x):
        _end = i + x
        if _end > len(data):
            _end = len(data)
        _chunk.append(data[i:_end])
    if int(len(data) / cpu_num) != len(data) / cpu_num:
        for i in range(x * count_x, len(data), y):
            _end = i + y
            if _end > len(data):
                _end = len(data)
            _chunk.append(data[i:_end])
    return _chunk

def reformatt_data(df):
    df_5 = pd.read_csv(f'/home/liyu/data/hedging-option/date_rate.csv', parse_dates=['date'])  #
    df_5.sort_values(by=['date'], ascending=False)
    for f in ['rate_1', 'rate_2', 'rate_3', 'rate_7', 'rate_14', 'rate_21']:
        prev_data_rate = 0
        for i in range(df_5.shape[0]):
            if df_5.loc[i, f] != df_5.loc[i, f]:  # check nan
                df_5.loc[i, f] = '-'
            if (df_5.loc[i, f]).strip() == '-' and prev_data_rate != 0:
                df_5.loc[i, f] = prev_data_rate
            df_5.loc[i, f] = df_5.loc[i, f] .strip()
            prev_data_rate = df_5.loc[i, f]
            if prev_data_rate == '-':
                print('error')
    for rate_i in ['rate_1', 'rate_2', 'rate_3', 'rate_7', 'rate_14', 'rate_21']:
        _rate = df_5[rate_i].to_numpy()
        percent_to_float = np.vectorize(lambda x: float(x.strip().strip('%')))
        rate_i_formatted = percent_to_float(_rate)
        df_5[f'{rate_i}_formatted'] = rate_i_formatted
    # df = pd.read_csv(f'{DATA_HOME_PATH}/all_raw_data_c.csv', parse_dates=['TradingDate'])
    df = pd.merge(df, df_5, how='left', left_on='TradingDate', right_on='date')
    df_m = pd.DataFrame(columns=df.columns)
    # df['date'] = df['TradingDate']
    trading_date = df.sort_values(by=['TradingDate'], ascending=False)['TradingDate'].unique()
    prev_date = None
    for i in tqdm(trading_date, total=len(trading_date)):
        d_data = df[df['TradingDate'] == i]
        if d_data.iloc[0]['date'] != d_data.iloc[0]['date'] and prev_date is not None:
            prev_d_data = df_m[df_m['TradingDate'] == prev_date].iloc[0]
            d_data.loc[:, 'date'] = i
            d_data.loc[:, 'rate_1_formatted'] = prev_d_data['rate_1_formatted']
            d_data.loc[:, 'rate_2_formatted'] = prev_d_data['rate_2_formatted']
            d_data.loc[:, 'rate_3_formatted'] = prev_d_data['rate_3_formatted']
            d_data.loc[:, 'rate_7_formatted'] = prev_d_data['rate_7_formatted']
            d_data.loc[:, 'rate_14_formatted'] = prev_d_data['rate_14_formatted']
            d_data.loc[:, 'rate_21_formatted'] = prev_d_data['rate_21_formatted']
            df_m = df_m.append(d_data)
        else:
            df_m = df_m.append(d_data)
        prev_date = i

    df = df_m
    df.drop(columns=['date','rate_1', 'rate_2', 'rate_3', 'rate_7', 'rate_14', 'rate_21'], axis=1, inplace=True)
    # df.to_csv(f'{DATA_HOME_PATH}/all_raw_data_c_formatted.csv', index=False)
    return df


def sub_time(df):
    mask_2020 = (df['TradingDate'] < pd.Timestamp('2023-01-01')) & (df['TradingDate'] > pd.Timestamp('2019-12-31'))
    df_year_2020_2022 = df[mask_2020]
    return df_year_2020_2022


def sub_type(df,type_n='C'):
    print(df.columns.to_numpy())
    df = df[df['CallOrPut']==type_n]
    return df


def my_log(func):
    """
    装饰器，用于打印日志
    """

    def inner(*args):
        print('\n', func.__name__, 'start ! args:', [i for i in args])
        func(*args)
        print(func.__name__, 'done !', '\n')

    return inner
