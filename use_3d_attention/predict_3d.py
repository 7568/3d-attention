from attention_3d import SELF_ATTENTION
from utils import logger_conf
from utils.load_data_3d import load_data
from utils.parser import get_parser, get_given_parameters_parser
from utils.scorer import get_scorer, BinScorer
from utils.timer import Timer


def training_validation_testing(model, X, y, training_trading_dates, validation_trading_dates, testing_trading_dates,
                                args, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)

    test_timer = Timer()

    # for i, (train_index, test_index) in enumerate(kf.split(X, y)):

    X_train, X_validation, X_test = X['training'], X['validation'], X['testing']
    y_train, y_validation, y_test = y['training'], y['validation'], y['testing']

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

    # Create a new unfitted version of the model
    curr_model = model.clone()


    test_timer.start()
    curr_model.set_testing(X_test, y_test, testing_trading_dates)
    max_day=210
    if args.dataset =='h_sh_300_option':
        max_day=360
    curr_model.predict_and_anasys_result(X_test, testing_trading_dates,max_day=max_day)
    test_timer.end()
    # sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)

    # print(sc.get_results())


def main_once(args):
    print("Train model with given hyperparameters")
    X, y, training_trading_dates, validation_trading_dates, testing_trading_dates = load_data(args)

    model_name = SELF_ATTENTION

    parameters = args.parameters[args.dataset][args.model_name]
    model = model_name(parameters, args)
    training_validation_testing(model, X, y, training_trading_dates, validation_trading_dates, testing_trading_dates,
                                args)


# python train.py --config config/h_sh_300_options.yml  --model_name LightGBM --n_trials 2 --epochs 30 --log_to_file &
if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    if arguments.log_to_file:
        logger_conf.init_log(f'{arguments.log_to_file_name}')
    print(arguments)
    parser = get_given_parameters_parser()
    arguments = parser.parse_args()
    arguments.gpu_index = 7
    main_once(arguments)
