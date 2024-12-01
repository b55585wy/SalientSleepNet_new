import numpy as np


def load_npz_file(npz_file: str) -> (np.ndarray, np.ndarray, int):
    """
    从npz文件加载数据。

    :param npz_file: npz文件名的字符串

    :return: 包含PSG数据、标签和采样率的元组
    """
    try:
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
    except Exception as e:
        raise Exception(f"加载文件 {npz_file} 时出错: {str(e)}")

    return data, labels, sampling_rate


def load_npz_files(npz_files: list) -> (list, list):
    """
    加载用于训练和验证的数据和标签

    :param npz_files: npz文件名的字符串列表

    :return: 数据列表和标签列表
    """
    data_list = []
    labels_list = []
    fs = None

    for npz_f in npz_files:
        print(f"正在加载 {npz_f} ...")
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)

        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception(f"发现采样率不匹配: {fs} != {sampling_rate}。")

        # 将数据扩展为适应Conv2d层的形状: [None, W, C, 1, 1] -> [C, None, W, H, N]
        tmp_data = np.squeeze(tmp_data)  # 形状为 [None, W, C]
        tmp_data = tmp_data[:, :, :, np.newaxis, np.newaxis]  # 增加维度 -> [None, W, C, H, N]

        # 如果有多个通道，合并成一个维度
        tmp_data = np.concatenate([tmp_data] * 3, axis=0)  # 扩展数据，变为 [C, None, W, H, N]

        # 转换数据类型
        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)

        data_list.append(tmp_data)
        labels_list.append(tmp_labels)

    print(f"总共加载了 {len(data_list)} 个文件。")
    return data_list, labels_list
