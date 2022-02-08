import time
from matplotlib import pyplot as plt
import tensorflow as tf
from src.container import ExperimentLogging, Model


class Trainer:
    def __init__(
        self,
        generator: Model,
        discriminator: Model,
        logging: ExperimentLogging,
    ):
        self.generator = generator.network
        self.discriminator = discriminator.network
        self.generator_loss = generator.loss
        self.discriminator_loss = discriminator.loss
        self.generator_optimizer = generator.optimizer
        self.discriminator_optimizer = discriminator.optimizer
        self.summary_writer = logging.summary_writer
        self.checkpoint_dir = logging.checkpoint_dir
        self.checkpoint_prefix = logging.checkpoint_prefix

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator.optimizer,
            discriminator_optimizer=discriminator.optimizer,
            generator=generator.network,
            discriminator=discriminator.network,
        )

    def fit(self, train_ds, test_ds, steps):
        example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if step % 1000 == 0:
                if step != 0:
                    print(f" Time taken for 1000 steps: {time.time() - start:.2f} sec\n")

                start = time.time()

                self.generate_images(
                    self.generator, example_input, example_target, step
                )
                print(f"Step: {step // 1000}k")

            self.train_step(input_image, target, step)

            if (step + 1) % 10 == 0:
                print(".", end="", flush=True)

            if (step + 1) % 5000 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    @tf.function
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator(
                [input_image, gen_output], training=True
            )

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
                disc_generated_output, gen_output, target
            )
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        with self.summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=step // 1000)
            tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step // 1000)
            tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step // 1000)
            tf.summary.scalar("disc_loss", disc_loss, step=step // 1000)

    @staticmethod
    def generate_images(model, test_input, tar, step):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ["Input Image", "Ground Truth", "Predicted Image"]

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])

            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis("off")

        plt.savefig(f"logs/samples/sample_{step}.png")
        plt.close()
