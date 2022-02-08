import tensorflow as tf


class DownSampleLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        size,
        apply_batchnorm=True,
    ):
        super().__init__()
        self.initializer = tf.random_normal_initializer(0.0, 0.02)
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(
                filters,
                size,
                strides=2,
                padding="same",
                kernel_initializer=self.initializer,
                use_bias=False,
            )
        )
        if apply_batchnorm:
            self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.LeakyReLU())

    def call(self, inputs):
        return self.model(inputs)


class UpSampleLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        size,
        apply_dropout=True,
    ):
        super().__init__()
        self.initializer = tf.random_normal_initializer(0.0, 0.02)
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Conv2DTranspose(
                filters,
                size,
                strides=2,
                padding="same",
                kernel_initializer=self.initializer,
                use_bias=False,
            )
        )
        self.model.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.ReLU())

    def call(self, inputs):
        return self.model(inputs)
