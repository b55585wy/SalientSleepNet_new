from tensorflow.keras import layers, models
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from construct_model import create_u_encoder, create_mse, upsample

class SingleSalientModel(models.Model):
    def __init__(self, padding: str ='same', build: bool = True, **kwargs):
        super(SingleSalientModel, self).__init__()

        # 初始化模型参数
        self.padding = padding
        self.sleep_epoch_length = kwargs['sleep_epoch_len']
        self.sequence_length = kwargs['preprocess']['sequence_epochs']
        self.filters = kwargs['train']['filters']
        self.kernel_size = kwargs['train']['kernel_size']
        self.pooling_sizes = kwargs['train']['pooling_sizes']
        self.dilation_sizes = kwargs['train']['dilation_sizes']
        self.activation = kwargs['train']['activation']
        self.u_depths = kwargs['train']['u_depths']
        self.u_inner_filter = kwargs['train']['u_inner_filter']
        self.mse_filters = kwargs['train']['mse_filters']

        if build:
            super().__init__(*self.init_model())

    def init_model(self, input: KerasTensor = None) -> (list, list):
        if input is None:
            input = layers.Input(shape=(self.sequence_length * self.sleep_epoch_length, 1, 1))

        l_name = "single_model_enc"

        # 编码器部分（5个编码器块）
        # encoder 1 [None, 60000, 1, 1] -> [None, 6000, 1, 8]
        u1 = create_u_encoder(input, self.filters[0], self.kernel_size, self.pooling_sizes[0],
                              middle_layer_filter=self.u_inner_filter, depth=self.u_depths[0],
                              pre_name=l_name, idx=1, padding=self.padding, activation=self.activation)
        u1 = layers.Conv2D(int(u1.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_layer_1",
                           padding=self.padding, activation=self.activation)(u1)
        pool = layers.MaxPooling2D((self.pooling_sizes[0], 1), name=f"{l_name}_pool1")(u1)

        # encoder 2 [None, 6000, 1, 8] -> [None, 750, 1, 16]
        u2 = create_u_encoder(pool, self.filters[1], self.kernel_size, self.pooling_sizes[1],
                              middle_layer_filter=self.u_inner_filter, depth=self.u_depths[1],
                              pre_name=l_name, idx=2, padding=self.padding, activation=self.activation)
        u2 = layers.Conv2D(int(u2.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_layer_2",
                           padding=self.padding, activation=self.activation)(u2)
        pool = layers.MaxPooling2D((self.pooling_sizes[1], 1), name=f"{l_name}_pool2")(u2)

        # encoder 3 [None, 750, 1, 16] -> [None, 125, 1, 32]
        u3 = create_u_encoder(pool, self.filters[2], self.kernel_size, self.pooling_sizes[2],
                              middle_layer_filter=self.u_inner_filter, depth=self.u_depths[2],
                              pre_name=l_name, idx=3, padding=self.padding, activation=self.activation)
        u3 = layers.Conv2D(int(u3.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_layer_3",
                           padding=self.padding, activation=self.activation)(u3)
        pool = layers.MaxPooling2D((self.pooling_sizes[2], 1), name=f"{l_name}_pool3")(u3)

        # encoder 4 [None, 125, 1, 32] -> [None, 50, 1, 64]
        u4 = create_u_encoder(pool, self.filters[3], self.kernel_size, self.pooling_sizes[3],
                              middle_layer_filter=self.u_inner_filter, depth=self.u_depths[3],
                              pre_name=l_name, idx=4, padding=self.padding, activation=self.activation)
        u4 = layers.Conv2D(int(u4.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_layer_4",
                           padding=self.padding, activation=self.activation)(u4)
        pool = layers.MaxPooling2D((self.pooling_sizes[3], 1), name=f"{l_name}_pool4")(u4)

        # encoder 5 [None, 50, 1, 64] -> [None, 50, 1, 128]
        u5 = create_u_encoder(pool, self.filters[4], self.kernel_size, self.pooling_sizes[4],
                              middle_layer_filter=self.u_inner_filter, depth=self.u_depths[4],
                              pre_name=l_name, idx=5, padding=self.padding, activation=self.activation)
        u5 = layers.Conv2D(int(u5.get_shape()[-1] * 0.5), (1, 1),
                           name=f"{l_name}_reduce_dim_layer_5",
                           padding=self.padding, activation=self.activation)(u5)
        pool = layers.MaxPooling2D((self.pooling_sizes[4], 1), name=f"{l_name}_pool5")(u5)

        # 解码器部分（例如可以通过上采样恢复到原始大小）
        decoder = upsample(pool, self.filters[-1], self.kernel_size, self.u_depths[-1])

        # 输出层
        output = layers.Dense(1, activation='sigmoid')(decoder)

        return [input], [output]
