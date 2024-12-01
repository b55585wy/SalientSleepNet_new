import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf


def weighted_categorical_cross_entropy(weights: np.ndarray):
    """
    Keras版本的加权分类交叉熵

    :param weights: 每个类别的权重列表

    :return: 加权分类交叉熵损失函数

    使用方法:
        weights = np.array([0.5,2,10])  # 权重分别为0.5、2、10
        loss = weighted_categorical_cross_entropy(weights)
        model.compile(loss=loss, optimizer='adam')
    """

    # 将权重转换为TensorFlow变量
    weights = tf.constant(weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # 缩放预测值，使每个样本的类别概率和为1
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)

        # 裁剪预测值，以防止出现NaN和Inf
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # 计算加权损失
        loss = y_true * tf.math.log(y_pred) * weights

        # 对所有类别求和并取负数
        loss = -tf.reduce_sum(loss, axis=-1)
        return loss

    return loss_fn
