import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import numpy as np
import pandas as pd


def reformat_data(training_df, validation_df, testing_df, not_use_pre_data=False):
    """
    :param training_df:
    :param validation_df:
    :param testing_df:
    :param not_use_pre_data:
    :return:
    """
    training_df.drop(columns=['TradingDate'], axis=1, inplace=True)
    validation_df.drop(columns=['TradingDate'], axis=1, inplace=True)
    testing_df.drop(columns=['TradingDate'], axis=1, inplace=True)
    target_fea = 'ClosePrice'
    train_x = training_df.copy()
    print(training_df.columns)
    train_x[target_fea]=1
    train_y = training_df[target_fea]

    validation_x = validation_df.copy()
    validation_x[target_fea]=1
    validation_y = validation_df[target_fea]

    testing_x = testing_df.copy()
    testing_x[target_fea]=1
    testing_y = testing_df[target_fea]

    # latest_x = latest_df.copy()
    # latest_x.loc[:, target_fea] = -1
    # latest_y = latest_df[target_fea]
    if not_use_pre_data:
        train_x = train_x.iloc[:, :int(train_x.shape[1] / 5)]
        validation_x = validation_x.iloc[:, :int(validation_x.shape[1] / 5)]
        testing_x = testing_x.iloc[:, :int(testing_x.shape[1] / 5)]
        # latest_x = latest_x.iloc[:, :int(latest_x.shape[1] / 5)]
    # cat_features = ['CallOrPut', 'MainSign']
    # for i in range(1, 5):
    #     cat_features.append(f'CallOrPut_{i}')
    #     cat_features.append(f'MainSign_{i}')
    # for f in cat_features:
    #     print(f'{f} : {testing_df.columns.get_loc(f)}')
    return train_x, train_y, validation_x, validation_y, testing_x, testing_y



def load_3_d_data(use_much_features,prepare_home_path,normal_type):

    training_df = pd.read_csv(f'{prepare_home_path}/{normal_type}/training.csv')
    validation_df = pd.read_csv(f'{prepare_home_path}/{normal_type}/validation.csv')
    testing_df = pd.read_csv(f'{prepare_home_path}/{normal_type}/testing.csv')
    less_features = ['ClosePrice', 'TradingDate','rate_7_formatted', 'UnderlyingScrtClose', 'ImpliedVolatility', 'StrikePrice',
                     'RemainingTerm']
    cat_features=[]
    if use_much_features:
        less_features = ['ClosePrice','TradingDate','TheoreticalPrice', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho',
                         'OpenPrice', 'HighPrice', 'LowPrice', 'SettlePrice', 'Change1', 'Change2', 'Volume',
                         'Position', 'Amount', 'PositionChange',
                         'rate_1_formatted', 'rate_2_formatted', 'rate_3_formatted', 'rate_7_formatted',
                         'rate_14_formatted', 'rate_21_formatted',
                         'AvgPrice', 'ClosePriceChangeRatio', 'SettlePriceChangeRatio', 'Amplitude', 'LimitUp',
                         'LimitDown',
                         'MaintainingMargin', 'ChangeRatio', 'UnderlyingScrtClose', 'ImpliedVolatility',
                         'StrikePrice', 'RemainingTerm','MainSign']
        cat_features = ['MainSign']
        for i in range(1, 5):
            cat_features.append(f'MainSign_{i}')
    _features = less_features.copy()
    _features.remove('TradingDate')
    for i in range(1, 5):
        for f in _features:
            less_features.append(f'{f}_{i}')
    training_df = training_df[less_features]
    validation_df = validation_df[less_features]
    testing_df = testing_df[less_features]
    sequence_length = 5
    features_n = len(less_features)//sequence_length
    training_trading_dates = training_df['TradingDate'].copy()
    validation_trading_dates = validation_df['TradingDate'].copy()
    testing_trading_dates = testing_df['TradingDate'].copy()
    train_x, train_y, validation_x, validation_y, testing_x, testing_y = reformat_data(
        training_df, validation_df, testing_df, not_use_pre_data=False)
    X = {
        'training': train_x.to_numpy(),
        'validation': validation_x.to_numpy(),
        'testing': testing_x.to_numpy(),
    }
    y = {
        'training': train_y.to_numpy(),
        'validation': validation_y.to_numpy(),
        'testing': testing_y.to_numpy(),
    }
    return X, y, training_trading_dates, validation_trading_dates, testing_trading_dates


def load_data(args):
    print("Loading dataset " + args.dataset + "...")
    X, y, training_trading_dates, validation_trading_dates, testing_trading_dates = load_3_d_data(args.use_much_features,args.prepare_home_path,args.normal_type)
    # # Preprocess target
    # if args.target_encode:
    #     le = LabelEncoder()
    #     y['training'] = le.fit_transform(y['training'])
    #     y['validation'] = le.fit_transform(y['validation'])
    #     y['testing'] = le.fit_transform(y['testing'])
    #
    #     # Setting this if classification task
    #     if args.objective == "classification":
    #         args.num_classes = len(le.classes_)
    #         print("Having", args.num_classes, "classes as target.")
    #
    # num_idx = []
    # args.cat_dims = []

    # Preprocess data
    # for i in range(args.num_features):
    #     if args.cat_idx and i in args.cat_idx:
    #         le = LabelEncoder()
    #         X['training'][:, i] = le.fit_transform(X['training'][:, i])
    #         X['validation'][:, i] = le.fit_transform(X['validation'][:, i])
    #         X['testing'][:, i] = le.fit_transform(X['testing'][:, i])
    #
    #         # Setting this?
    #         if len(le.classes_) == 1:
    #             args.cat_dims.append(len(le.classes_) + 1)
    #         else:
    #             args.cat_dims.append(len(le.classes_))
    #
    #     else:
    #         num_idx.append(i)
    #
    # if args.scale:
    #     print("Scaling the data...")
    #     scaler = StandardScaler()
    #     X['training'][:, num_idx] = scaler.fit_transform(X['training'][:, num_idx])
    #     X['validation'][:, num_idx] = scaler.fit_transform(X['validation'][:, num_idx])
    #     X['testing'][:, num_idx] = scaler.fit_transform(X['testing'][:, num_idx])
    #
    # if args.one_hot_encode:
    #     ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    #     new_x1 = ohe.fit_transform(X[:, args.cat_idx])
    #     new_x2 = X[:, num_idx]
    #     X = np.concatenate([new_x1, new_x2], axis=1)
    #     print("New Shape:", X.shape)
    return X, y, training_trading_dates, validation_trading_dates, testing_trading_dates



