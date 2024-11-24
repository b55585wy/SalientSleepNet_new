import numpy as np
import tensorflow.keras.backend as K


def weighted_categorical_cross_entropy(weights: np.ndarray):
    """
    keras.objectives.categorical_crossentropy的加权版本

    :param weights: 每个类别的权重列表

    :return: 一个加权的分类交叉熵损失函数

    使用方法:
        weights = np.array([0.5,2,10]) # 第一类权重0.5，第二类权重2倍，第三类权重10倍。
        loss = weighted_categorical_cross_entropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    # 将权重转换为Keras变量
    weights = K.variable(weights)

    def loss_fn(y_true, y_pred):
        # 缩放预测值，使每个样本的类别概率和为1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        
        # 裁剪预测值以防止出现NaN和Inf
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # 计算加权损失
        loss = y_true * K.log(y_pred) * weights
        
        # 对所有类别求和得到最终损失
        loss = -K.sum(loss, -1)
        return loss

    return loss_fn
