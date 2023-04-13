import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from openpyxl import Workbook


def create_table_001():
    df = pd.read_csv(f'{DATA_HOME_PATH}/all_raw_data_c_formatted.csv', parse_dates=['TradingDate'])
    print(df.shape)
    moneyness_2017 = df[df['TradingDate'] < pd.Timestamp('2018-01-01')]
    mask_2018 = (df['TradingDate'] < pd.Timestamp('2019-01-01')) & (df['TradingDate'] > pd.Timestamp('2017-12-31'))
    moneyness_2018 = df[mask_2018]
    mask_2019 = (df['TradingDate'] < pd.Timestamp('2020-01-01')) & (df['TradingDate'] > pd.Timestamp('2018-12-31'))
    moneyness_2019 = df[mask_2019]
    mask_2020 = (df['TradingDate'] < pd.Timestamp('2021-01-01')) & (df['TradingDate'] > pd.Timestamp('2019-12-31'))
    moneyness_2020 = df[mask_2020]
    mask_2021 = (df['TradingDate'] < pd.Timestamp('2022-01-01')) & (df['TradingDate'] > pd.Timestamp('2020-12-31'))
    moneyness_2021 = df[mask_2021]
    mask_2021 = (df['TradingDate'] < pd.Timestamp('2023-01-01')) & (df['TradingDate'] > pd.Timestamp('2021-12-31'))
    moneyness_2022 = df[mask_2021]
    wb = Workbook()
    ws = wb.active
    ws.append(['Year', '', 'r', 'spot', 'vol', 'strike_price', 'maturity', 'c'])
    date = ['2017', '2018', '2019', '2020', '2021', '2022']
    for index_0, m in enumerate(
            [moneyness_2017, moneyness_2018, moneyness_2019, moneyness_2020, moneyness_2021, moneyness_2022]):
        r = m['rate_7_formatted'].to_numpy()
        spot = m['UnderlyingScrtClose'].to_numpy()
        vol = m['HistoricalVolatility'].to_numpy()
        strike_price = m['StrikePrice'].to_numpy()
        maturity = m['RemainingTerm'].to_numpy()
        c = m['ClosePrice'].to_numpy()
        for n in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            ws.append([date[index_0], n, '', '', '', '', '', ''])
        for index_1, s in enumerate([r, spot, vol, strike_price, maturity, c]):
            count = len(s)
            mean = s.mean()
            std = s.std()
            min = s.min()
            _25 = np.percentile(s, 27)
            _50 = np.percentile(s, 50)
            _50_1 = np.median(s)
            # print(_50==_50_1)
            _75 = np.percentile(s, 75)
            max = s.max()
            for index_2, v in enumerate([count, mean, std, min, _25, _50, _75, max]):
                ws.cell(row=(index_0 * 8 + 2 + index_2), column=index_1 + 3).value = v
    wb.save('output.xlsx')





def reformatt_data():
    df_5 = pd.read_csv(f'/home/liyu/data/hedging-option/date_rate.csv', parse_dates=['date'])  #
    df_5.sort_values(by=['date'], ascending=False)
    for f in ['rate_1', 'rate_2', 'rate_3', 'rate_7', 'rate_14', 'rate_21']:
        prev_data_rate = 0
        for i in range(df_5.shape[0]):
            if df_5.loc[i, f] != df_5.loc[i, f]:  # check nan
                df_5.loc[i, f] = '-'
            if (df_5.loc[i, f]).strip() == '-' and prev_data_rate != 0:
                df_5.loc[i, f] = prev_data_rate
            prev_data_rate = df_5.loc[i, f]
            if prev_data_rate == '-':
                print('error')
    rate_7 = df_5['rate_7'].to_numpy()
    percent_to_float = np.vectorize(lambda x: float(x.strip().strip('%')))
    rate_7_formatted = percent_to_float(rate_7)
    df_5['rate_7_formatted'] = rate_7_formatted
    df = pd.read_csv(f'{DATA_HOME_PATH}/all_raw_data_c.csv', parse_dates=['TradingDate'])
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
            # d_data.iloc[:, 'rate_1'] = prev_d_data['rate_1']
            # d_data.iloc[:, 'rate_2'] = prev_d_data['rate_2']
            # d_data.iloc[:, 'rate_3'] = prev_d_data['rate_3']
            # d_data.iloc[:, 'rate_7'] = prev_d_data['rate_7']
            # d_data.iloc[:, 'rate_14'] = prev_d_data['rate_14']
            # d_data.iloc[:, 'rate_21'] = prev_d_data['rate_21']
            d_data.loc[:, 'rate_7_formatted'] = prev_d_data['rate_7_formatted']
            df_m = df_m.append(d_data)
        else:
            df_m = df_m.append(d_data)
        prev_date = i

    df = df_m
    df.to_csv(f'{DATA_HOME_PATH}/all_raw_data_c_formatted.csv', index=False)

DATA_HOME_PATH = f'/home/liyu/data/hedging-option/20170101-20230101/'
# OPTION_SYMBOL = 'index-option/h_sh_300'
OPTION_SYMBOL='ETF50-option'


def create_table_002():
    create_table_001()


if __name__ == '__main__':
    DATA_HOME_PATH = DATA_HOME_PATH + "/" + OPTION_SYMBOL + "/"
    reformatt_data()
    # create_table_001()
    
    create_table_002()

