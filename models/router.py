import tensorflow as tf


class LongShortTermRouter(tf.keras.layers.Layer):
    def __init__(self, hidden_size: int):
        super(LongShortTermRouter, self).__init__()
        # Gate层：全连接层 + Sigmoid 激活
        self.gate = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size),
            tf.keras.layers.Activation('sigmoid')
        ])

    def call(self, global_features, local_features):
        """
        Args:
            global_features (tf.Tensor): [batch_size, seq_len, hidden_size]
            local_features (tf.Tensor): [batch_size, seq_len, hidden_size]
        Returns:
            tf.Tensor: [batch_size, seq_len, hidden_size]
        """
        # 计算重要性权重
        combined = tf.concat([global_features, local_features], axis=-1)
        importance = self.gate(combined)

        # 动态融合
        output = importance * global_features + (1 - importance) * local_features
        return output
