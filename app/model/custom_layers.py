from keras.saving import register_keras_serializable
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.models import load_model

@register_keras_serializable()
@register_keras_serializable()
class Avg2MaxPooling(layers.Layer):
    def __init__(self, pool_size=3, strides=2, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.avg_pool = layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)
        self.max_pool = layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
        self.bn = layers.BatchNormalization()

    def call(self, inputs):
        x = self.avg_pool(inputs) - 2 * self.max_pool(inputs)
        return self.bn(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding
        })
        return config



@register_keras_serializable()
class SEBlock(layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(units=max(1, self.channels // self.ratio), activation='swish')
        self.fc2 = layers.Dense(units=self.channels, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, self.channels))

    def call(self, inputs):
        se = self.global_pool(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)
        return inputs * se

@register_keras_serializable()
class DepthwiseSeparableConv(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, se_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.dw = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same')
        self.pw = layers.Conv2D(filters=filters, kernel_size=1)
        self.bn = layers.BatchNormalization()
        self.se = SEBlock(se_ratio)
        self.proj = layers.Conv2D(filters=filters, kernel_size=1) if strides != 1 else None

    def call(self, inputs):
        residual = inputs
        x = self.dw(inputs)
        x = self.pw(x)
        x = self.bn(x)
        x = tf.nn.swish(x)
        x = self.se(x)
        if self.proj is not None:
            residual = self.proj(residual)
        return x + residual if residual.shape == x.shape else x

