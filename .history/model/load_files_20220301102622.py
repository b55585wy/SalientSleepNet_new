import numpy as np


def load_npz_file(npz_file: str) -> (np.ndarray, np.ndarray, int):
    """
    Load data from npz files.

    :param npz_file: a str of npz filename

    :return: a tuple of PSG data, labels and sampling rate of the npz file
    """
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate


def load_npz_files(npz_files: list) -> (list, list):
    """
    Load data and labels for training and validation

    :param npz_files: a list of str for npz file names

    :return: the lists of data and labels
    """
    data_list = []
    labels_list = []
    fs = None

    for npz_f in npz_files:
        print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")

        # We add one extra axis for adaptation the Conv2d layer
        # Here N denote filter number, C denote the channel number,
        # W denote the width(Sleep Epoch Length), H denote the height
        tmp_data = np.squeeze(tmp_data)  # the shape is [None, W, C]
        tmp_data = tmp_data[:, :, :, np.newaxis, np.newaxis]  # expaned the N and H axis, [None, W, C, H, N]
        tmp_data = np.concatenate((tmp_data[np.newaxis, :, :, 0, :, :], tmp_data[np.newaxis, :, :, 1, :, :],
                                   tmp_data[np.newaxis, :, :, 2, :, :]), axis=0)  # transmute the shape to [C, None, W, H, N]


        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)

        data_list.append(tmp_data)
        labels_list.append(tmp_labels)

    print(f"load {len(data_list)} files totally.")

    return data_list, labels_list
