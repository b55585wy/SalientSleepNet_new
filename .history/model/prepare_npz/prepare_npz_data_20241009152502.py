import argparse
import datetime
import glob
import math
import ntpath
import os
import shutil

import numpy as np
import pandas
from mne.io import read_raw_edf

from edf_header_reader import BaseEDFReader

# 定义睡眠阶段的数字编码
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

# 睡眠阶段字符串到数字的映射
stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

# 数字到睡眠阶段字符串的映射
class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

# 注释到标签的映射
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30  # 每个epoch的秒数

def main():
    """将EDF+文件转换为npz文件的主函数。"""
    # 准备命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="../sleep_data/sleepedf-39",
                        help="包含睡眠信息的EDF文件路径。")
    parser.add_argument("--output_dir", "-o", type=str, default="../sleep_data/sleepedf/prepared",
                        help="保存输出的目录。")
    parser.add_argument("--select_ch", '-s', type=list, default=["EEG Fpz-Cz", "EOG horizontal", "EMG submental"],
                        help="选择用于训练的通道。")
    args = parser.parse_args()

    # 创建或清空输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # 选择的通道
    select_ch: list = args.select_ch

    # 读取原始和注释EDF文件
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i, file in enumerate(psg_fnames):
        # 读取EDF文件
        raw = read_raw_edf(file, preload=True, stim_channel=None)
        sampling_rate = raw.info['sfreq']
        raw_ch_df = raw.to_data_frame(scaling_time=100.0)[select_ch]
        if not isinstance(raw_ch_df, pandas.DataFrame):
            raw_ch_df = raw_ch_df.to_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))

        # 获取原始文件头信息
        with open(file, 'r', encoding='iso-8859-1') as f:
            reader_raw = BaseEDFReader(f)
            reader_raw.read_header()
            h_raw = reader_raw.header
        raw_start_dt = datetime.datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

        # 读取注释文件及其头信息
        with open(ann_fnames[i], 'r', encoding='iso-8859-1') as f:
            reader_ann = BaseEDFReader(f)
            reader_ann.read_header()
            h_ann = reader_ann.header
            _, _, ann = list(zip(*reader_ann.records()))
        ann_start_dt = datetime.datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

        # 确保原始文件和注释文件的开始时间相同
        assert raw_start_dt == ann_start_dt

        # 生成标签并移除不需要的索引
        remove_idx = []  # 将被移除的数据索引
        labels = []  # 有标签的数据索引
        label_idx = []
        for a in ann[0]:
            onset_sec, duration_sec, ann_char = a
            ann_str = "".join(ann_char)
            label = ann2label[ann_str]
            if label != UNKNOWN:
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise Exception("Something wrong")
                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                label_epoch = np.ones(duration_epoch, dtype=np.int) * label
                labels.append(label_epoch)
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
                label_idx.append(idx)

                print(f"Include onset:{onset_sec}, duration:{duration_sec}, label:{label}, ({ann_str})")
            else:
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int)
                remove_idx.append(idx)

                print(f"Remove onset:{onset_sec}, duration:{duration_sec}, label:{label}, ({ann_str})")
        labels = np.hstack(labels)

        # 移除不需要的数据
        print(f'before remove unwanted: {np.arange(len(raw_ch_df)).shape}')
        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)
            select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
        else:
            select_idx = np.arange(len(raw_ch_df))
        print(f"after remove unwanted: {select_idx.shape}")

        # 只选择有标签的数据
        print(f"before intersect label: {select_idx.shape}")
        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)
        print(f"after intersect label: {select_idx.shape}")

        # 移除多余的索引
        if len(label_idx) > len(select_idx):
            print(f"before remove extra labels: {select_idx.shape}, {labels.shape}")
            extra_idx = np.setdiff1d(label_idx, select_idx)
            # 修剪尾部
            if np.all(extra_idx > select_idx[-1]):
                n_label_trims = int(math.ceil(len(extra_idx) / (EPOCH_SEC_SIZE * sampling_rate)))
                if n_label_trims != 0: labels = labels[:-n_label_trims]
            print(f"after remove extra labels: {select_idx.shape}, {labels.shape}")

        # 移除运动和未知阶段（如果有的话）
        raw_ch = raw_ch_df.values[select_idx]

        # 验证是否可以分割为30秒的epoch
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("Something wrong")
        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

        # 获取epoch及其对应的标签
        x = np.asarray(np.split(raw_ch, int(n_epochs))).astype(np.float32)
        y = labels.astype(np.int32)

        assert len(x) == len(y)

        # 只选择睡眠期
        w_edge_min = 30
        nw_idx = np.where(y != stage_dict['W'])[0]
        start_idx = nw_idx[0] - (w_edge_min * 2)
        end_idx = nw_idx[-1] + (w_edge_min * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        print(f"Data before selection: {x.shape}, {y.shape}")
        x = x[select_idx]
        y = y[select_idx]
        print(f"Data after selection: {x.shape}, {y.shape}")

        # 保存数据
        filename = ntpath.basename(file).replace("-PSG.edf", ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "ch_label": select_ch,
            "header_raw": h_raw,
            "header_annotation": h_ann,
        }

        np.savez_compressed(os.path.join(args.output_dir, filename), **save_dict)  # 压缩保存

        print("\n=======================================\n")

if __name__ == '__main__':
    main()