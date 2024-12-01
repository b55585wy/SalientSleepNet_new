import numpy as np
from concurrent.futures import ThreadPoolExecutor


def normalization(data: np.ndarray) -> np.ndarray:
    """
    对PSG数据进行归一化处理

    :param data: PSG数据
    :return: 归一化后的PSG数据
    """
    for i in range(data.shape[0]):
        data[i] -= data[i].mean(axis=0)  # 减去均值
        data[i] /= data[i].std(axis=0)  # 除以标准差
    return data


def preprocess(data: list, labels: list, param: dict, not_enhance: bool = False) -> (np.ndarray, np.ndarray):
    """
    预处理原始PSG数据,将其转换为可以输入模型的序列

    :param data: PSG数据列表
    :param labels: 睡眠阶段标签列表
    :param param: 超参数字典
    :param not_enhance: 是否使用数据增强,默认为False(使用增强)
    :return: 处理后的数据序列和标签数组
    """

    def data_big_group(d: np.ndarray) -> np.ndarray:
        """
        将数据分割成大组,以防止数据增强时的数据泄露
        """
        num_groups = d.shape[1] // param['big_group_size']
        return_data = np.zeros((d.shape[0], num_groups, param['big_group_size'], d.shape[2]))
        for i in range(num_groups):
            return_data[:, i, :, :] = d[:, i * param['big_group_size']:(i + 1) * param['big_group_size'], :]
        return return_data

    def label_big_group(l: np.ndarray) -> np.ndarray:
        """
        将标签分割成大组,以防止数据增强时的数据泄露
        """
        # 这里与 data_big_group 类似，返回处理后的标签数据
        num_groups = l.shape[1] // param['big_group_size']
        return_data = np.zeros((l.shape[0], num_groups, param['big_group_size']))
        for i in range(num_groups):
            return_data[:, i, :] = l[:, i * param['big_group_size']:(i + 1) * param['big_group_size']]
        return return_data

    def data_window_slice(d: np.ndarray) -> np.ndarray:
        """
        应用数据增强
        """
        stride = param['sequence_epochs'] if not not_enhance else param['enhance_window_stride']
        num_samples = (d.shape[2] - param['sequence_epochs']) // stride + 1
        return_data = np.zeros((d.shape[0], num_samples, param['sequence_epochs'], d.shape[1]))

        for i in range(num_samples):
            return_data[:, i, :, :] = d[:, :, i * stride:i * stride + param['sequence_epochs']]
        return return_data

    def labels_window_slice(l: np.ndarray) -> np.ndarray:
        """
        对标签应用数据增强
        """
        stride = param['sequence_epochs'] if not not_enhance else param['enhance_window_stride']
        num_samples = (l.shape[1] - param['sequence_epochs']) // stride + 1
        return_data = np.zeros((l.shape[0], num_samples, param['sequence_epochs']))

        for i in range(num_samples):
            return_data[:, i, :] = l[:, i * stride:i * stride + param['sequence_epochs']]
        return return_data

    # 使用并行化加速数据预处理
    with ThreadPoolExecutor() as executor:
        preprocessed_data = list(executor.map(data_window_slice, data))
        preprocessed_labels = list(executor.map(labels_window_slice, labels))

    # 合并数据和标签
    processed_data = np.concatenate(preprocessed_data, axis=0)
    processed_labels = np.concatenate(preprocessed_labels, axis=0)

    return processed_data, processed_labels
