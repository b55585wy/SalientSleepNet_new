import numpy as np


def load_npz_file(npz_file: str) -> (np.ndarray, np.ndarray, int):
    """
    从npz文件加载数据。

    :param npz_file: npz文件名的字符串

    :return: 包含PSG数据、标签和采样率的元组
    """
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
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
        print("正在加载 {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("发现采样率不匹配。")

        # 我们添加一个额外的轴以适应Conv2d层
        # 这里N表示滤波器数量，C表示通道数，
        # W表示宽度（睡眠周期长度），H表示高度
        tmp_data = np.squeeze(tmp_data)  # 形状为 [None, W, C]
        tmp_data = tmp_data[:, :, :, np.newaxis, np.newaxis]  # 扩展N和H轴，形状变为 [None, W, C, H, N]
        tmp_data = np.concatenate((tmp_data[np.newaxis, :, :, 0, :, :], 
                                   tmp_data[np.newaxis, :, :, 1, :, :],
                                   tmp_data[np.newaxis, :, :, 2, :, :]), axis=0)  # 转换形状为 [C, None, W, H, N]

        # 将数据类型转换为float32和int32
        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)

        data_list.append(tmp_data)
        labels_list.append(tmp_labels)

    print(f"总共加载了 {len(data_list)} 个文件。")

    return data_list, labels_list
