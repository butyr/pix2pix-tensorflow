import tensorflow as tf
from src.layers import DownSampleLayer, UpSampleLayer


class UNet(tf.keras.Model):
    def __init__(self, num_channels):
        super().__init__()

        self.num_channels = num_channels
        self.downsample_stack = [
            DownSampleLayer(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            DownSampleLayer(128, 4),  # (batch_size, 64, 64, 128)
            DownSampleLayer(256, 4),  # (batch_size, 32, 32, 256)
            DownSampleLayer(512, 4),  # (batch_size, 16, 16, 512)
            DownSampleLayer(512, 4),  # (batch_size, 8, 8, 512)
            DownSampleLayer(512, 4),  # (batch_size, 4, 4, 512)
            DownSampleLayer(512, 4),  # (batch_size, 2, 2, 512)
            DownSampleLayer(512, 4),  # (batch_size, 1, 1, 512)
        ]
        self.upsample_stack = [
            UpSampleLayer(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            UpSampleLayer(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            UpSampleLayer(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            UpSampleLayer(512, 4),  # (batch_size, 16, 16, 1024)
            UpSampleLayer(256, 4),  # (batch_size, 32, 32, 512)
            UpSampleLayer(128, 4),  # (batch_size, 64, 64, 256)
            UpSampleLayer(64, 4),  # (batch_size, 128, 128, 128)
        ]
        self.last = tf.keras.layers.Conv2DTranspose(
            self.num_channels,
            4,
            strides=2,
            padding="same",
            kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
            activation="tanh",
        )  # (batch_size, 256, 256, 3)

    def call(self, inputs):
        x, skip_connections = self._downsample(inputs)
        x = self._upsample(x, skip_connections)

        return self.last(x)

    def _downsample(self, x):
        skip_connections = []

        for down in self.downsample_stack:
            x = down(x)
            skip_connections.append(x)

        return x, skip_connections

    def _upsample(self, x, skip_connections):
        skip_connections = reversed(skip_connections[:-1])

        for up, skip in zip(self.upsample_stack, skip_connections):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        return x


class PatchGAN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        initializer = tf.random_normal_initializer(0.0, 0.02)

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=[512, 256, 3], name="merged_input"),
                DownSampleLayer(64, 4, apply_batchnorm=False),
                DownSampleLayer(128, 4),
                DownSampleLayer(256, 4),
                tf.keras.layers.ZeroPadding2D(),
                tf.keras.layers.Conv2D(
                    512, 4, strides=1, kernel_initializer=initializer, use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.ZeroPadding2D(),
                tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer),
            ]
        )

    def call(self, inputs):
        inputs_merged = tf.concat(inputs, axis=1)
        return self.model(inputs_merged)
