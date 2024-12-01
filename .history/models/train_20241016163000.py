import os
import re
import glob
import shutil
import logging
import argparse
import itertools

import yaml
import numpy as np
import tensorflow.keras.models
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical, multi_gpu_model

from preprocess import preprocess
from load_files import load_npz_files
from evaluation import draw_training_plot
from models import SingleSalientModel, TwoSteamSalientModel
from loss_function import weighted_categorical_cross_entropy


def gpu_settings():
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)


def get_parser() -> argparse.Namespace:
    """
    parser arguments and setting log formats
    :return: the arguments after parse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", '-g', default='1', help="the number of gpus for training.")
    parser.add_argument("--modal", '-m', default='1',
                        help="the way to training model\n\t0: single modal.\n\tmulti modal.")
    parser.add_argument("--data_dir", '-d', default="../sleep_data/sleepedf/prepared-cassette", help="where the data is.")
    parser.add_argument("--output_dir", '-o', default='./result', help="where you wish to set the results.")
    parser.add_argument("--valid", '-v', default='20', help="v stands for k-fold validation's k.")
    parser.add_argument("--from_fold", default='0', help="the first fold to train.")
    parser.add_argument("--train_fold", default='5', help="the folds you want to train this time.")

    args = parser.parse_args()

    res_path = args.output_dir
    if os.path.exists(res_path):
        shutil.rmtree(res_path)
    os.makedirs(res_path)

    # log settings
    logging.basicConfig(filemode='a', filename=f'{res_path}/log.log', level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] in %(funcName)s - %(levelname)s: %(message)s')

    k_folds = eval(args.valid)
    if not isinstance(k_folds, int):
        logging.critical("the argument type `valid` should be an integer")
        print("ERROR: get an invalid `k_fold`")
        exit(-1)
    if k_folds <= 0:
        logging.critical(f"get an invalid `k_folds`: {k_folds}")
        print(f"ERROR: the `k_fold` should be positive, but get: {k_folds}")
        exit(-1)

    from_fold = eval(args.from_fold)
    if not isinstance(from_fold, int):
        logging.critical("the argument `from_fold` should be an integer")
        print("ERROR: get an invalid type `from_fold`")
        exit(-1)
    if not (0 <= from_fold <= k_folds):
        logging.critical(f"get an invalid `from_fold`: {from_fold}")
        print(f"ERROR: the `from_fold` should between 0 and {k_folds}, but get {from_fold}")
        exit(-1)

    train_fold = eval(args.train_fold)
    if not isinstance(train_fold, int):
        logging.critical("the argument `train_fold` should be an integer")
        print("ERROR: get an invalid type `train_fold`")
        exit(-1)
    if train_fold <= 0:
        logging.critical(f"get an invalid `train_fold`: {train_fold}")
        print(f"ERROR: the `train_fold` should greater than 0, but get {train_fold}")
        exit(-1)

    modal = eval(args.modal)
    if not isinstance(modal, int):
        logging.critical("the argument `modal` ought to be an integer")
        print("ERROR: get an invalid type `modal`")
        exit(-1)
    if modal != 1 and modal != 0:
        logging.critical(f"get an invalid `modal`: {modal}")
        print(f"ERROR: the `modal` ought to between 0 and 1, but get {modal}")
        exit(-1)

    return args


def print_params(params: dict):
    """
    a function to formatted print model's hyperparameters
    :param params: a dict contain all hyperparameters
    """
    print("=" * 20, "[Hyperparameters]", "=" * 20)
    for (key, val) in params.items():
        if isinstance(val, dict):
            print(f"{key}:")
            for (k, v) in val.items():
                print(f"\t{k}: {v}")
        else:
            print(f"{key}: {val}")
    print("=" * 60)


def train(args: argparse.Namespace, hyper_param_dict: dict) -> dict:
    """
    a function to training the salient sleep net model
    :param args: the argument from command line input
    :param hyper_param_dict: a dict contain model's hyper parameters
    :return: a dict contain the training history
    """

    # fetch arguments
    res_path = args.output_dir
    k_folds = eval(args.valid)
    from_fold = eval(args.from_fold)
    train_fold = eval(args.train_fold)
    if from_fold + train_fold > k_folds:
        train_fold = k_folds - from_fold
    modal = eval(args.modal)

    # fetch gpu numbers
    gpu_num = eval(args.gpus) if 1 <= eval(args.gpus) <= 4 else 1
    if gpu_num == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    elif gpu_num == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    elif gpu_num == 3:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'
    elif gpu_num == 4:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    logging.info(f"gpu numbers: {gpu_num}")

    # load data
    npz_names = glob.glob(os.path.join(args.data_dir, '*.npz'))
    if len(npz_names) == 0:
        logging.critical(f"Can not find any npz file in {args.data_dir}")
        print("ERROR: Failed to load data")
        exit(-1)
    npz_names.sort()

    npzs_list = []
    ids = 20 if len(npz_names) < 100 else 83  # 20 for sleepedf-39, 83 for sleepedf-153
    for id in range(ids):
        inner_list = []
        for name in npz_names:
            pattern = re.compile(f".*SC4{id:02}[12][EFG]0\.npz")
            if re.match(pattern, name):
                inner_list.append(name)
        if inner_list:  # not empty
            npzs_list.append(inner_list)
    npzs_list = np.asarray(npzs_list)
    npzs_list = np.array_split(npzs_list, k_folds)

    # save the split for summary
    save_dict = {'split': npzs_list}
    np.savez(os.path.join(res_path, 'split.npz'), **save_dict)

    sleep_epoch_len = hyper_param_dict['sleep_epoch_len']

    # loss function
    weighted_loss = weighted_categorical_cross_entropy(np.asarray(hyper_param_dict['class_weights']))
    print(f"loss weights: {hyper_param_dict['class_weights']}")

    # result
    acc_list, val_acc_list = [], []
    loss_list, val_loss_list = [], []

    if modal == 0:
        model: tensorflow.keras.models.Model = SingleSalientModel(**hyper_param_dict)
    else:
        model: tensorflow.keras.models.Model = TwoSteamSalientModel(**hyper_param_dict)

    model.summary()
    if gpu_num > 1:
        model = multi_gpu_model(model, gpus=gpu_num)

    model.compile(optimizer=hyper_param_dict["optimizer"], loss=weighted_loss, metrics=['acc'])
    model.save_weights('weights.h5')
    # k fold training and validation
    for i in range(from_fold, from_fold + train_fold):
        logging.info(f"began to validation: {i + 1}/{k_folds}")
        print(f"{k_folds}-validation, turn: {i + 1}")

        valid_npzs = list(itertools.chain.from_iterable(npzs_list[i].tolist()))
        train_npzs = list(set(npz_names) - set(valid_npzs))

        logging.info("begin to load data")
        train_data_list, train_labels_list = load_npz_files(train_npzs)
        val_data_list, val_labels_list = load_npz_files(valid_npzs)
        logging.info("data loaded")

        train_labels_list = [to_categorical(f) for f in train_labels_list]
        val_labels_list = [to_categorical(f) for f in val_labels_list]

        epochnumber = 0  # Calculate epoch number
        for i1 in train_labels_list:
            epochnumber += i1.shape[0]
        for i1 in val_labels_list:
            epochnumber += i1.shape[0]
        print("Total epoch number is " + str(epochnumber))

        logging.info("begin to preprocess")  # [C, None, group_seq_len * W, H, N]
        train_data, train_labels = preprocess(train_data_list, train_labels_list, hyper_param_dict['preprocess'], True)
        val_data, val_labels = preprocess(val_data_list, val_labels_list, hyper_param_dict['preprocess'], True)
        logging.info("preprocess down")

        logging.info("begin to shuffle training data and labels")
        index = [i for i in range(train_data.shape[1])]
        np.random.shuffle(index)
        for j in range(len(train_data)):
            train_data[j] = train_data[j][index]
        train_labels = train_labels[index]
        logging.info("shuffle completed")

        print(f"train on {train_data.shape[1]} samples, each has {train_data.shape[2] / sleep_epoch_len} sleep epochs")
        print(f"validate on {val_data.shape[1]} samples, each has {val_data.shape[2] / sleep_epoch_len} sleep epochs")

        logging.info(f"begin to train fold {i+1}...")

        callback_list = [
            callbacks.EarlyStopping(monitor='acc', patience=hyper_param_dict['patience']),
            callbacks.ModelCheckpoint(filepath=os.path.join(res_path, f"fold_{i + 1}_best_model.h5"),
                                      monitor='val_acc', save_best_only=True, save_weights_only=True)
        ]

        if modal == 0:  # only use EEG, the shape is [None, W * gsl, H, N]
            history = model.fit(train_data[0], train_labels, epochs=hyper_param_dict['train']['epochs'],
                                batch_size=hyper_param_dict['train']['batch_size'], callbacks=callback_list,
                                validation_data=(val_data[0], val_labels), verbose=2)
        elif modal == 1:  # use EEG & EOG, the shape is [C, None, W * gsl, H, N]
            history = model.fit([train_data[0], train_data[1]], train_labels, epochs=hyper_param_dict['train']['epochs'],
                                batch_size=hyper_param_dict['train']['batch_size'], callbacks=callback_list,
                                validation_data=([val_data[0], val_data[1]], val_labels), verbose=2)
        logging.info(f"fold {i+1} completed.")

        acc_list.append(history.history['acc'])
        val_acc_list.append(history.history['val_acc'])
        loss_list.append(history.history['loss'])
        val_loss_list.append(history.history['val_loss'])

        K.clear_session()

        # clear weights
        model.reset_states()
        model.load_weights('weights.h5')

    res_dict = {
        'acc': acc_list,
        'val_acc': val_acc_list,
        'loss': loss_list,
        'val_loss': val_loss_list,
    }
    return res_dict


if __name__ == "__main__":
    gpu_settings()

    args = get_parser()

    with open("./hyperparameters.yaml", encoding='utf-8') as f:
        hyper_params = yaml.full_load(f)
    print_params(hyper_params)

    train_history = train(args, hyper_params)
    draw_training_plot(train_history, eval(args.from_fold) + 1, eval(args.train_fold), args.output_dir)
