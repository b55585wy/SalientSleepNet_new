import numpy as np
from concurrent.futures import ThreadPoolExecutor


def normalization(data: np.ndarray) -> np.ndarray:
    """
    :param data: the PSG data

    :return: PSG data after normalization
    """
    for i in range(data.shape[0]):
        data[i] -= data[i].mean(axis=0)
        data[i] /= data[i].std(axis=0)
    return data


def preprocess(data: list, labels: list, param: dict, not_enhance: bool = False) -> (np.ndarray, np.ndarray):
    """
    To preprocess the raw PSG data into sequence that can feed into the model

    :param data: the list of PSG data
    :param labels: the list of sleep stage labels
    :param param: a dict of hyperparameter
    :param not_enhance: decide whether to use data enhancement

    :return: the array of sequence of data and labels
    """

    def data_big_group(d: np.ndarray) -> np.ndarray:
        """
        A closure to divide data into big groups to prevent data leak in data enhancement
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
        A closure to divide labels into big groups to prevent data leak in data enhancement
        """
        return_labels = np.array([])
        beg = 0
        while (beg + param['big_group_size']) <= len(l):
            y = l[beg: beg + param['big_group_size']]
            y = y[np.newaxis, ...]
            return_labels = y if beg == 0 else np.append(return_labels, y, axis=0)
            beg += param['big_group_size']
        return return_labels

    def data_window_slice(d: np.ndarray) -> np.ndarray:
        """
        A closure to apply data enhancement
        """

        # we don't apply data enhancement if it for validation
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
        A closure to apply data enhancement for labels
        """
        stride = param['sequence_epochs'] if not_enhance else param['enhance_window_stride']

        return_labels = np.array([])
        for cnt1, group in enumerate(l):
            flat_labels = np.array([])
            cnt2 = 0
            while (cnt2 + param['sequence_epochs']) <= len(group):
                y = np.vstack(group[cnt2: cnt2 + param['sequence_epochs']])
                y = y[np.newaxis, ...]
                flat_labels = y if cnt2 == 0 else np.append(flat_labels, y, axis=0)
                cnt2 += stride
            return_labels = flat_labels if cnt1 == 0 else np.append(return_labels, flat_labels, axis=0)
        return return_labels

    # create a threads pool to process every item of the lists
    data_executor = ThreadPoolExecutor(max_workers=8)
    after_regular_data = data_executor.map(normalization, data)
    after_divide_data = data_executor.map(data_big_group, after_regular_data)
    after_enhance_data = data_executor.map(data_window_slice, after_divide_data)
    after_divide_labels = data_executor.map(label_big_group, labels)
    after_enhance_labels = data_executor.map(labels_window_slice, after_divide_labels)
    data_executor.shutdown()

    final_data = []
    final_labels = []
    for ind, dt in enumerate(after_enhance_data):
        final_data = dt if ind == 0 else np.append(final_data, dt, axis=1)
    for ind, lb in enumerate(after_enhance_labels):
        final_labels = lb if ind == 0 else np.append(final_labels, lb, axis=0)

    return final_data, final_labels[:, :, np.newaxis, :]


if __name__ == "__main__":
    from load_files import load_npz_files
    import yaml
    import glob
    import os
    with open("hyperparameters.yaml", encoding='utf-8') as f:
        hyper_params = yaml.full_load(f)
    data, labels = load_npz_files(glob.glob(os.path.join(r'D:\Python\MySleepProject\sleep_data\sleepedf-39', '*.npz')))
    data, labels = preprocess(data, labels, hyper_params['preprocess'])
    pass
