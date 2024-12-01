import os
import glob
import logging
import argparse
import numpy as np
import tensorflow as tf
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
    parser.add_argument("--modal", '-m', default='1', help="训练模型的方式\n\t0: 单模态\n\t1: 多模态")
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

    # 评估模型并计算性能
    for model_name in model_names:
        model = tf.keras.models.load_model(model_name)  # 加载模型
        # 假设你有预定义的评估函数
        metrics = evaluate_model(model, npz_names, res_dir)
        print(f"Model {model_name}: {metrics}")


def evaluate_model(model, npz_names, res_dir):
    """
    评估模型
    :param model: 训练好的TensorFlow模型
    :param npz_names: 数据集划分信息
    :param res_dir: 结果目录
    :return: 模型评估指标
    """
    all_preds = []
    all_labels = []

    # 假设你加载数据并处理
    for npz_file in npz_names:
        data = np.load(npz_file)
        signals, labels = data['signals'], data['labels']

        # 模型预测
        predictions = model.predict(signals)
        preds = np.argmax(predictions, axis=-1)  # 获取预测的类别

        # 存储真实标签和预测结果
        all_labels.extend(labels)
        all_preds.extend(preds)

    # 计算评价指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes=["Class1", "Class2", "Class3"], title="Confusion Matrix")

    return {"accuracy": accuracy, "f1_score": f1}


if __name__ == "__main__":
    # 解析参数
    args = parse_args()

    # 读取模型超参数配置
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    summary_models(args, config)
