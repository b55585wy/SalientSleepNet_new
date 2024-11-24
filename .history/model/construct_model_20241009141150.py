from tensorflow.keras import layers
from tensorflow.python.keras.engine.keras_tensor import KerasTensor


def upsample(dst: KerasTensor, pre_name: str = '', idx: int = 0) -> layers.Layer:
    """
    使用双线性插值的自定义上采样层
    """
    from tensorflow import image
    import numpy as np
    np.random.rand(30)
    return layers.Lambda(lambda x, w, h: image.resize(x, (w, h)),
                         arguments={'w': dst.shape[1], 'h': dst.shape[2]}, name=f"{pre_name}_upsample{idx}")


def create_bn_conv(input: KerasTensor, filter: int, kernel_size: int, dilation_rate: int = 1, pre_name: str = '',
                   idx: int = 0, padding='same', activation: str = 'relu') -> KerasTensor:
    """
    创建一个基本的卷积-批归一化结构
    """
    conv = layers.Conv2D(filter, (kernel_size, 1), padding=padding,
                         dilation_rate=(dilation_rate, dilation_rate), activation=activation,
                         name=f"{pre_name}_conv{idx}")(input)
    bn = layers.BatchNormalization(name=f"{pre_name}_bn{idx}")(conv)
    return bn


def create_u_encoder(input: KerasTensor, filter: int, kernel_size: int, pooling_size: int,
                     middle_layer_filter: int, depth: int, pre_name: str = '', idx: int = 0,
                     padding: str = 'same', activation: str = 'relu') -> KerasTensor:
    """
    创建U型编码器，SalientSleepNet的重复子结构
    """
    l_name = f"{pre_name}_U{idx}_enc"
    from_encoder = []
    conv_bn0 = create_bn_conv(input, filter, kernel_size,
                             pre_name=l_name, idx=0,
                             padding=padding, activation=activation)
    conv_bn = conv_bn0
    for d in range(depth - 1):
        conv_bn = create_bn_conv(conv_bn, middle_layer_filter, kernel_size,
                                 pre_name=l_name, idx=d + 1,
                                 padding=padding, activation=activation)
        from_encoder.append(conv_bn)
        if d != depth - 2:
            conv_bn = layers.MaxPooling2D((pooling_size, 1), name=f"{l_name}_pool{d + 1}")(conv_bn)

    conv_bn = create_bn_conv(conv_bn, middle_layer_filter, kernel_size,
                             pre_name=l_name, idx=depth,
                             padding=padding, activation=activation)

    l_name = f"{pre_name}_U{idx}_dec"
    for d in range(depth - 1, 0, -1):
        conv_bn = upsample(from_encoder[-1], pre_name=l_name, idx=d)(conv_bn)
        ch = filter if d == 1 else middle_layer_filter
        conv_bn = create_bn_conv(layers.concatenate([conv_bn, from_encoder.pop()]),
                                 ch, kernel_size, pre_name=l_name,
                                 idx=d, padding=padding, activation=activation)

    return layers.add([conv_bn, conv_bn0])


def create_mse(input: KerasTensor, filter: int, kernel_size: int, dilation_rates: list, pre_name: str = "",
               idx: int = 0, padding: str = 'same', activation: str = "relu") -> KerasTensor:
    """
    创建多尺度提取模块，SalientSleepNet的重复子结构
    """
    l_name = f"{pre_name}_mse{idx}"

    convs = []
    for (i, dr) in enumerate(dilation_rates):
        conv_bn = create_bn_conv(input, filter, kernel_size, dilation_rate=dr,
                                 pre_name=l_name, idx=1 + i, padding=padding, activation=activation)
        convs.append(conv_bn)

    from functools import reduce
    con_conv = reduce(lambda l, r: layers.concatenate([l, r]), convs)

    down = layers.Conv2D(filter * 2, (kernel_size, 1), name=f"{l_name}_downconv1",
                         padding=padding, activation=activation)(con_conv)
    down = layers.Conv2D(filter, (kernel_size, 1), name=f"{l_name}_downconv2",
                         padding=padding, activation=activation)(down)
    out = layers.BatchNormalization(name=f"{l_name}_bn{len(dilation_rates) + 1}")(down)

    return out
