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
        data[i] /= data[i].std(axis=0)   # 除以标准差
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
        return_data = np.array([])
        beg = 0
        while (beg + param['big_group_size']) <= d.shape[1]:
            y = d[:, beg: beg + param['big_group_size'], ...]
            y = y[:, np.newaxis, ...]
            return_data = y if beg == 0 else np.append(return_data, y, axis=1)
            beg += param['big_group_size']
        return return_data

    def label_big_group(l: np.ndarray) -> np.ndarray:
        """
        将标签分割成大组,以防止数据增强时的数据泄露
        """
        # ... (省略相似代码)

    def data_window_slice(d: np.ndarray) -> np.ndarray:
        """
        应用数据增强
        """
        # 如果是验证集,则不使用数据增强
        stride = param['sequence_epochs'] if not_enhance else param['enhance_window_stride']

        return_data = np.array([])
        for cnt1, modal in enumerate(d):
            modal_data = np.array([])
            for cnt2, group in enumerate(modal):
                flat_data = np.array([])
                cnt3 = 0
                while (cnt3 + param['sequence_epochs']) <= len(group):
                    y = np.vstack(group[cnt3: cnt3 + param['sequence_epochs']])
                    y = y[np.newaxis, ...]
                    flat_data = y if cnt3 == 0 else np.append(flat_data, y, axis=0)
                    cnt3 += stride
                modal_data = flat_data if cnt2 == 0 else np.append(modal_data, flat_data, axis=0)
            modal_data = modal_data[np.newaxis, ...]
            return_data = modal_data if cnt1 == 0 else np.append(return_data, modal_data, axis=0)
        return return_data

    def labels_window_slice(l: np.ndarray) -> np.ndarray:
        """
        对标签应用数据增强
        """
        # ... (省略相似代码)

    # 创建线程池来处理列表中的每个项目
    data_executor = ThreadPoolExecutor(max_workers=8)
    after_regular_data = data_executor.map(normalization, data)  # 对数据进行归一化
    after_divide_data = data_executor.map(data_big_group, after_regular_data)  # 将数据分割成大组
    after_enhance_data = data_executor.map(data_window_slice, after_divide_data)  # 应用数据增强
    after_divide_labels = data_executor.map(label_big_group, labels)  # 将标签分割成大组
    after_enhance_labels = data_executor.map(labels_window_slice, after_divide_labels)  # 对标签应用数据增强
    data_executor.shutdown()

    # 合并处理后的数据和标签
    final_data = []
    final_labels = []
    for ind, dt in enumerate(after_enhance_data):
        final_data = dt if ind == 0 else np.append(final_data, dt, axis=1)
    for ind, lb in enumerate(after_enhance_labels):
        final_labels = lb if ind == 0 else np.append(final_labels, lb, axis=0)

    return final_data, final_labels[:, :, np.newaxis, :]


if __name__ == "__main__":
    # 主函数,用于测试预处理功能
    from load_files import load_npz_files
    import yaml
    import glob
    import os
    
    # 加载超参数
    with open("hyperparameters.yaml", encoding='utf-8') as f:
        hyper_params = yaml.full_load(f)
    
    # 加载数据文件
    data, labels = load_npz_files(glob.glob(os.path.join(r'C:\Users\a1396\Documents\GitHub\SalientSleepNet\model\sleep_data\sleepedf\prepared', '*.npz')))
    
    # 预处理数据
    data, labels = preprocess(data, labels, hyper_params['preprocess'])
    pass