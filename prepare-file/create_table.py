import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from openpyxl import Workbook


def create_table_001(ta):
    df = pd.read_csv(f'{DATA_HOME_PATH}/all_raw_data_c_formatted.csv', parse_dates=['TradingDate'])
    print(df.shape)
    df_year_2017 = df[df['TradingDate'] < pd.Timestamp('2018-01-01')]
    mask_2018 = (df['TradingDate'] < pd.Timestamp('2019-01-01')) & (df['TradingDate'] > pd.Timestamp('2017-12-31'))
    df_year_2018 = df[mask_2018]
    mask_2019 = (df['TradingDate'] < pd.Timestamp('2020-01-01')) & (df['TradingDate'] > pd.Timestamp('2018-12-31'))
    df_year_2019 = df[mask_2019]
    mask_2020 = (df['TradingDate'] < pd.Timestamp('2021-01-01')) & (df['TradingDate'] > pd.Timestamp('2019-12-31'))
    df_year_2020 = df[mask_2020]
    mask_2021 = (df['TradingDate'] < pd.Timestamp('2022-01-01')) & (df['TradingDate'] > pd.Timestamp('2020-12-31'))
    df_year_2021 = df[mask_2021]
    mask_2021 = (df['TradingDate'] < pd.Timestamp('2023-01-01')) & (df['TradingDate'] > pd.Timestamp('2021-12-31'))
    df_year_2022 = df[mask_2021]
    print(df.shape[0])
    print(df_year_2017.shape[0] + df_year_2018.shape[0] + df_year_2019.shape[0] + df_year_2020.shape[0] + df_year_2021.shape[0] + df_year_2022.shape[0])
    save_data_statistics([ '2020', '2021', '2022'],[df_year_2020,df_year_2021,df_year_2022],ta)





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
    for rate_i in ['rate_1', 'rate_2', 'rate_3', 'rate_7', 'rate_14', 'rate_21']:
        _rate = df_5[rate_i].to_numpy()
        percent_to_float = np.vectorize(lambda x: float(x.strip().strip('%')))
        rate_i_formatted = percent_to_float(_rate)
        df_5[f'{rate_i}_formatted'] = rate_i_formatted
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



def caculate_df_year(df):
    # moneyness = spot/strike
    spot = df['UnderlyingScrtClose'].to_numpy()
    strike = df['StrikePrice'].to_numpy()
    df_year = spot/strike
    df.loc[:, 'moneyness'] = df_year
    return df


def save_data_statistics(ta,tb,tc):
    wb = Workbook()
    ws = wb.active
    ws.append(['Year', '', 'r', 'spot', 'vol', 'strike_price', 'maturity', 'c'])
    date = ta
    for index_0, m in enumerate(tb):
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
            mean = "{:.4g}".format(s.mean())
            std = "{:.4g}".format(s.std())
            min = "{:.4g}".format(s.min())
            _25 = "{:.4g}".format(np.percentile(s, 27))
            _50 = "{:.4g}".format(np.percentile(s, 50))
            _50_1 = "{:.4g}".format(np.median(s))
            # print(_50==_50_1)
            _75 = "{:.4g}".format(np.percentile(s, 75))
            max = "{:.4g}".format(s.max())
            for index_2, v in enumerate([count, mean, std, min, _25, _50, _75, max]):
                ws.cell(row=(index_0 * 8 + 2 + index_2), column=index_1 + 3).value = v
    wb.save(f'{tc}.xlsx')


def create_table_002(ta):
    #H_sh_300 call option data statistics sorted by moneyness
    df = pd.read_csv(f'{DATA_HOME_PATH}/all_raw_data_c_formatted.csv', parse_dates=['TradingDate'])
    print(df.shape)
    df = caculate_df_year(df)

    mask_2020 = (df['TradingDate'] < pd.Timestamp('2023-01-01')) & (df['TradingDate'] > pd.Timestamp('2019-12-31'))
    df_year_2020_2022 = df[mask_2020]
    print(f'df_year_2020_2022.shape : {df_year_2020_2022.shape}')
    moneyness_less_097 = df_year_2020_2022[df_year_2020_2022['moneyness'] <= 0.97]
    print(f'moneyness_less_097.shape : {moneyness_less_097.shape}')
    moneyness_in_097_103 = df_year_2020_2022.loc[(df_year_2020_2022['moneyness'] > 0.97) & (df_year_2020_2022['moneyness'] <= 1.03)]
    print(f'moneyness_in_097_103.shape : {moneyness_in_097_103.shape}')
    moneyness_more_103 = df_year_2020_2022[df_year_2020_2022['moneyness'] > 1.03]
    print(f'moneyness_more_103.shape : {moneyness_more_103.shape}')
    print(f'sum is {moneyness_less_097.shape[0] + moneyness_in_097_103.shape[0] + moneyness_more_103.shape[0]} \n')
    save_data_statistics(['<0.97' , '0.97–1.03' , '≥1.03'],[moneyness_less_097,moneyness_in_097_103,moneyness_more_103],ta)



HOME_PATH = f'/home/liyu/data/hedging-option/20170101-20230101/'

if __name__ == '__main__':
    OPTION_SYMBOL = 'index-option/h_sh_300'
    DATA_HOME_PATH = HOME_PATH + "/" + OPTION_SYMBOL + "/"
    # reformatt_data()
    create_table_001('h_sh_300_sorted_by_years')

    OPTION_SYMBOL = 'ETF50-option'
    DATA_HOME_PATH = HOME_PATH + "/" + OPTION_SYMBOL + "/"
    # reformatt_data()
    create_table_001('ETF50_sorted_by_years')

    OPTION_SYMBOL = 'index-option/h_sh_300'
    DATA_HOME_PATH = HOME_PATH + "/" + OPTION_SYMBOL + "/"
    create_table_002('h_sh_300_sorted_by_moneyness')

    OPTION_SYMBOL = 'ETF50-option'
    DATA_HOME_PATH = HOME_PATH + "/" + OPTION_SYMBOL + "/"
    create_table_002('ETF50_sorted_by_moneyness')

