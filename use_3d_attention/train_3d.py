import logging
import sys

import optuna

from utils import logger_conf
from utils.load_data_3d import load_data
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import save_results_to_file, save_hyperparameters_to_file, save_loss_to_file
from utils.parser import get_parser, get_given_parameters_parser
from attention_3d import SELF_ATTENTION
from sklearn.model_selection import KFold, StratifiedKFold  # , train_test_split


def training_validation_testing(model, X, y, training_trading_dates, validation_trading_dates, testing_trading_dates,
                                args, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()

    # for i, (train_index, test_index) in enumerate(kf.split(X, y)):

    X_train, X_validation, X_test = X['training'], X['validation'], X['testing']
    y_train, y_validation, y_test = y['training'], y['validation'], y['testing']

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

    # Create a new unfitted version of the model
    curr_model = model.clone()

    # Train model
    train_timer.start()
    curr_model.set_testing(X_test, y_test, testing_trading_dates)
    loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_validation, y_validation,
                                                    training_trading_dates, validation_trading_dates)  # X_val, y_val)
    # loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_test, y_test,training_trading_dates, testing_trading_dates)  # X_val, y_val)
    train_timer.end()

    test_timer.start()
    curr_model.predict(X_test, testing_trading_dates)
    test_timer.end()

    # Save model weights and the truth/prediction pairs for traceability
    # curr_model.save_model_and_predictions(y_test)

    if save_model:
        save_loss_to_file(args, loss_history, "loss")
        save_loss_to_file(args, val_loss_history, "val_loss")

    # Compute scores on the output
    sc.eval(curr_model.testing_y, curr_model.predictions, curr_model.prediction_probabilities)

    print(sc.get_results())

    # Best run is saved to file
    if save_model:
        print("Results:", sc.get_results())
        print("Train time:", train_timer.get_average_time())
        print("Inference time:", test_timer.get_average_time())

        # Save the all statistics to a file
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), test_timer.get_average_time(),
                             model.params)

    # print("Finished cross validation")
    return sc, (train_timer.get_average_time(), test_timer.get_average_time())



def main_once(args):
    print("Train model with given hyperparameters")

    args.PREPARE_HOME_PATH = args.dataset
    args.NORMAL_TYPE = args.normal_type
    X, y, training_trading_dates, validation_trading_dates, testing_trading_dates = load_data(args)

    model_name = SELF_ATTENTION

    parameters = args.parameters[args.dataset][args.model_name]
    model = model_name(parameters, args)
    sc, time = training_validation_testing(model, X, y, training_trading_dates, validation_trading_dates,
                                               testing_trading_dates, args)
    print('finished training model')
    print(sc.get_results())
    print(time)


# python train.py --config config/h_sh_300_options.yml  --model_name LightGBM --n_trials 2 --epochs 30 --log_to_file &
if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    # arguments.config = 'config/h_sh_300_options_3d_attention.yml'
    if arguments.log_to_file:
        logger_conf.init_log(f'{arguments.log_to_file_name}')
    print(arguments)
    # Also load the best parameters
    parser = get_given_parameters_parser()
    arguments = parser.parse_args()
    # update default gpu_index
    if arguments.gpu_index == 0:
        arguments.gpu_index = 7
    if arguments.gpu_index > 7:
        arguments.gpu_index %= 8
    main_once(arguments)


"""
h_sh_300
use_much_features: false
mse : 0.007881278172135353 , rmse : 0.08877656608819962 , mae : 0.06949970871210098
mse in validation : 0.007881278172135353
mse : 0.000903825624845922 , rmse : 0.0300636924803257 , mae : 0.01908879354596138
mse in testing : 0.000903825624845922


mse : 0.004092297051101923 , rmse : 0.06397106498479843 , mae : 0.03928695619106293
mse in testing : 0.004092297051101923

"""