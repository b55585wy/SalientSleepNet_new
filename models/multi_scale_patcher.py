import tensorflow as tf


class MultiScalePatcher(tf.keras.layers.Layer):
    def __init__(self, input_len, scales=[1, 2, 4, 8]):
        super(MultiScalePatcher, self).__init__()
        self.input_len = input_len
        self.scales = scales

    def call(self, signal):
        """
        Args:
            signal (tf.Tensor): [batch_size, channels, time_steps]
        Returns:
            list of tf.Tensor: 不同尺度的patches
        """
        patches = []
        for scale in self.scales:
            patch_size = self.input_len // scale
            # 使用 tf.image.extract_patches 来进行滑动窗口操作
            patch = tf.image.extract_patches(
                images=signal,  # 输入信号
                sizes=[1, 1, patch_size, 1],  # 窗口大小
                strides=[1, 1, patch_size, 1],  # 步长
                rates=[1, 1, 1, 1],  # 不做空间扩展
                padding="VALID"  # 无填充
            )
            # 需要重新排列为 [batch_size, channels, num_patches] 的形状
            patches.append(patch)
        return patches
