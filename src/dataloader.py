import tensorflow as tf


class Dataloader:
    def __init__(
            self,
            path,
            buffer_size: int,
            batch_size: int,
            img_width: int,
            img_height: int,
    ):
        self.path = path
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height

    def build_train(self):
        dataset = tf.data.Dataset.list_files(str(self.path / 'train/*.jpg'))
        dataset = dataset.map(self.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size)

        return dataset
    
    def build_test(self):
        try:
            dataset = tf.data.Dataset.list_files(str(self.path / 'test/*.jpg'))
        except tf.errors.InvalidArgumentError:
            dataset = tf.data.Dataset.list_files(str(self.path / 'val/*.jpg'))
        dataset = dataset.map(self.load_image_test)
        dataset = dataset.batch(self.batch_size)

        return dataset

    @staticmethod
    def load(image_file):
        image = tf.io.read_file(image_file)
        image = tf.io.decode_jpeg(image)

        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, w:, :]
        real_image = image[:, :w, :]

        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    @staticmethod
    def resize(input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image

    def random_crop(self, input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
          stacked_image, size=[2, self.img_height, self.img_width, 3])

        return cropped_image[0], cropped_image[1]

    @staticmethod
    def normalize(input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image

    @tf.function()
    def random_jitter(self, input_image, real_image):
        input_image, real_image = self.resize(input_image, real_image, 286, 286)
        input_image, real_image = self.random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def load_image_train(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.random_jitter(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image

    def load_image_test(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.resize(input_image, real_image, self.img_height, self.img_width)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image
