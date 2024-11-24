import os
import glob
import logging
import argparse
import itertools
from functools import reduce

import yaml
import numpy as np
import tensorflow.keras.models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from preprocess import preprocess
from load_files import load_npz_files
from loss_function import weighted_categorical_cross_entropy
from evaluation import f1_scores_from_cm, plot_confusion_matrix
from models import SingleSalientModel, TwoSteamSalientModel

def parse_args():
    """
    解析命令行参数并设置日志格式
    :return: 解析后的参数
    """
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument("--data_dir", "-d", default="./data/sleepedf-2013/npzs", help="数据所在目录")
    parser.add_argument("--modal", '-m', default='1',
                        help="训练模型的方式\n\t0: 单模态\n\t1: 多模态")
    parser.add_argument("--output_dir", '-o', default='./result', help="结果输出目录")
    parser.add_argument("--valid", '-v', default='20', help="k折交叉验证中的k值")

    args = parser.parse_args()

    # 验证k_folds参数
    k_folds = eval(args.valid)
    if not isinstance(k_folds, int):
        logging.critical("valid参数应为整数")
        print("错误: 获得了无效的k_fold值")
        exit(-1)
    if k_folds <= 0:
        logging.critical(f"获得了无效的k_folds值: {k_folds}")
        print(f"错误: k_fold应为正数,但获得了: {k_folds}")
        exit(-1)

    # 验证modal参数
    modal = eval(args.modal)
    if not isinstance(modal, int):
        logging.critical("modal参数应为整数")
        print("错误: 获得了无效的modal类型")
        exit(-1)
    if modal != 1 and modal != 0:
        logging.critical(f"获得了无效的modal值: {modal}")
        print(f"错误: modal应为0或1,但获得了{modal}")
        exit(-1)

    return args


def summary_models(args: argparse.Namespace, hyper_params: dict):
    """
    总结模型性能
    :param args: 命令行输入的参数
    :param hyper_params: 包含模型超参数的字典
    """
    modal = eval(args.modal)
    k_folds = eval(args.valid)
    res_dir = args.output_dir

    # 加载数据集划分信息
    with np.load(os.path.join(res_dir, "split.npz"), allow_pickle=True) as f:
        npz_names = f['split']

    # 获取所有模型文件名
    model_names = glob.glob(os.path.join(res_dir, "fold_*_best_model.h5"))
    if len(model_names) < k_folds:
        logging.critical(f"没有足够的模型进行总结,需要{k_folds}个但只有{len(model_names)}个")
        exit(-1)
    model_names.sort()

    # 定义损失函数
    loss = weighted_categorical_cross_entropy(hyper_params['class_weights'])

    best_turn_f1, best_turn_acc = 0.0, 0.0
    best_turn_name = ''
    cm_list = []

    # 根据modal参数选择模型
    if modal == 0:
        eva_model: tensorflow.keras.models.Model = SingleSalientModel(**hyper_params)
    else:
        eva_model: tensorflow.keras.models.Model = TwoSteamSalientModel(**hyper_params)

    eva_model.compile(optimizer=hyper_params['optimizer'], loss=loss, metrics=['acc'])

    # 对每个fold进行评估
    for i in range(k_folds):
        # 加载训练权重
        eva_model.load_weights(model_names[i])

        # 加载并处理测试数据
        test_npzs = list(itertools.chain.from_iterable(npz_names[i].tolist()))
        test_data_list, test_labels_list = load_npz_files(test_npzs)
        test_labels_list = [to_categorical(f) for f in test_labels_list]

        test_data, test_labels = preprocess(test_data_list, test_labels_list, hyper_params['preprocess'], True)

        logging.info(f"评估 {os.path.basename(model_names[i])} ,共 {test_data.shape[1]} 个样本")

        # 预测
        y_pred = np.array([])
        if modal == 0:
            y_pred: np.ndarray = eva_model.predict(test_data[0], batch_size=hyper_params['train']['batch_size'])
        elif modal == 1:
            y_pred: np.ndarray = eva_model.predict([test_data[0], test_data[1]], batch_size=hyper_params['train']['batch_size'])

        y_pred = y_pred.reshape((-1, 5))
        test_labels = test_labels.reshape((-1, 5))

        # 计算评估指标
        acc = accuracy_score(test_labels.argmax(axis=1), y_pred.argmax(axis=1))
        f1 = f1_score(test_labels.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
        cm = confusion_matrix(test_labels.argmax(axis=1), y_pred.argmax(axis=1))

        cm = np.array(cm)
        print(f"{os.path.basename(model_names[i])}的准确率为{acc:.4},F1分数为{f1:.4}")
        print("混淆矩阵:")
        print(cm.astype('float32') / np.sum(cm).astype('float32'))
        plot_confusion_matrix(cm, classes=hyper_params['evaluation']['label_class'], title=f"cm_{i+1}", path=res_dir)
        plot_confusion_matrix(cm, classes=hyper_params['evaluation']['label_class'],
                              normalize=False, title=f"cm_num_{i+1}", path=res_dir)

        if f1 > best_turn_f1:
            best_turn_f1, best_turn_acc = f1, acc
            best_turn_name = os.path.basename(model_names[i])

        cm_list.append(cm)
        logging.info(f"evaluate {os.path.basename(model_names[i])} completed.")
        eva_model.reset_states()

    print(f"the best model is {best_turn_name} with accuracy={best_turn_acc} and f1-score={best_turn_f1}")

    sum_cm = reduce(lambda x, y: x + y, cm_list)
    plot_confusion_matrix(sum_cm, classes=hyper_params['evaluation']['label_class'], title='cm_total', path=res_dir)
    plot_confusion_matrix(sum_cm, classes=hyper_params['evaluation']['label_class'], title='cm_total_num',
                          normalize=False,  path=res_dir)
    ave_f1 = f1_scores_from_cm(sum_cm)
    ave_acc = np.sum(np.diagonal(sum_cm)) / np.sum(sum_cm)
    print(f"the average accuracy: {ave_acc} and the average f1-score: {ave_f1}")


if __name__ == '__main__':
    args = parse_args()
    with open("hyperparameters.yaml", encoding='utf-8') as f:
        hyper_params = yaml.full_load(f)

    summary_models(args, hyper_params)
