import datetime
import logging
import re
from typing import TextIO as File

import numpy as np

EVENT_CHANNEL = 'EDF Annotations'
log = logging.getLogger(__name__)

# 自定义异常，用于表示EDF数据结束
class EDFEndOfData(BaseException):
    """自定义异常，用于表示EDF数据结束。"""
    pass


def tal(tal_str: str) -> list:
    """解析EDF+时间戳注释列表（TALs）。
    
    参数:
        tal_str (str): 要解析的TAL字符串。
    
    返回:
        list: 包含(开始时间, 持续时间, 注释)元组的列表。
    """
    # 用于解析TAL的正则表达式
    exp = r'(?P<onset>[+\-]\d+(?:\.\d*)?)' + \
          r'(?:\x15(?P<duration>\d+(?:\.\d*)?))?' + \
          r'(\x14(?P<annotation>[^\x00]*))?' + \
          r'(?:\x14\x00)'

    return [(
        float(dic['onset']),
        float(dic['duration']) if dic['duration'] else 0.,
        dic['annotation'].split('\x14') if dic['annotation'] else []
    )
        for dic
        in map(lambda m: m.groupdict(), re.finditer(exp, tal_str))
    ]


def edf_header(f: File) -> dict:
    """读取并解析EDF文件头。
    
    参数:
        f (File): EDF文件对象。
    
    返回:
        dict: 包含解析后的文件头信息的字典。
    """
    h = {}
    assert f.tell() == 0  # 检查文件位置
    assert f.read(8) == '0       '

    # 解析记录信息
    h['local_subject_id'] = f.read(80).strip()
    h['local_recording_id'] = f.read(80).strip()

    # 解析时间戳
    day, month, year = [int(x) for x in re.findall(r'(\d+)', f.read(8))]
    hour, minute, sec = [int(x) for x in re.findall(r'(\d+)', f.read(8))]
    h['date_time'] = str(datetime.datetime(year + 2000, month, day, hour, minute, sec))

    # 解析其他文件头信息
    header_ntypes = int(f.read(8))
    subtype = f.read(44)[:5]
    h['EDF+'] = subtype in ['EDF+C', 'EDF+D']
    h['contiguous'] = subtype != 'EDF+D'
    h['n_records'] = int(f.read(8))
    h['record_length'] = float(f.read(8))  # 单位：秒
    nchannels = h['n_channels'] = int(f.read(4))

    # 读取通道信息
    channels = list(range(h['n_channels']))
    h['label'] = [f.read(16).strip() for _ in channels]
    h['transducer_type'] = [f.read(80).strip() for _ in channels]
    h['units'] = [f.read(8).strip() for _ in channels]
    h['physical_min'] = np.asarray([float(f.read(8)) for _ in channels])
    h['physical_max'] = np.asarray([float(f.read(8)) for _ in channels])
    h['digital_min'] = np.asarray([float(f.read(8)) for _ in channels])
    h['digital_max'] = np.asarray([float(f.read(8)) for _ in channels])
    h['prefiltering'] = [f.read(80).strip() for _ in channels]
    h['n_samples_per_record'] = [int(f.read(8)) for _ in channels]
    f.read(32 * nchannels)  # 保留字段

    assert f.tell() == header_ntypes
    return h


class BaseEDFReader:
    """用于读取EDF文件的基类。"""

    def __init__(self, file: File):
        self.gain = None
        self.phys_min = None
        self.dig_min = None
        self.header = None
        self.file = file

    def read_header(self):
        """读取并处理EDF文件头。"""
        self.header = h = edf_header(self.file)

        # 计算重新缩放的范围
        self.dig_min = h['digital_min']
        self.phys_min = h['physical_min']
        phys_range = h['physical_max'] - h['physical_min']
        dig_range = h['digital_max'] - h['digital_min']
        assert np.all(phys_range > 0)
        assert np.all(dig_range > 0)
        self.gain = phys_range / dig_range

    def read_raw_record(self) -> list:
        """从EDF文件中读取原始数据记录。
        
        返回:
            list: 包含每个通道原始字节数组的列表。
        """
        result = []
        for nsamp in self.header['n_samples_per_record']:
            samples = self.file.read(nsamp * 2)
            if len(samples) != nsamp * 2:
                raise EDFEndOfData
            result.append(samples)
        return result

    def convert_record(self, raw_record: list) -> (float, list, list):
        """将原始记录转换为(时间, 信号, 事件)元组。
        
        参数:
            raw_record (list): 原始记录数据。
        
        返回:
            tuple: (时间, 信号列表, 事件列表)
        """
        h = self.header
        dig_min, phys_min, gain = self.dig_min, self.phys_min, self.gain
        time = float('nan')
        signals, events = [], []

        for i, samples in enumerate(raw_record):
            if h['label'][i] == EVENT_CHANNEL:
                ann = tal(samples)
                time = ann[0][0]
                events.extend(ann[1:])
            else:
                # 将2字节小端整数转换为物理值
                dig = np.fromstring(samples, '<i2').astype(np.float32)
                phys = (dig - dig_min[i]) * gain[i] + phys_min[i]
                signals.append(phys)

        return time, signals, events

    def read_record(self) -> (float, list, list):
        """读取并转换单个记录。"""
        return self.convert_record(self.read_raw_record())

    def records(self):
        """生成器，用于读取EDF文件中的所有记录。"""
        try:
            while True:
                yield self.read_record()
        except EDFEndOfData:
            pass