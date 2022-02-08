import tensorflow as tf


class GeneratorLoss:
    def __init__(
        self,
        lam,
    ):
        self.lam = lam
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def __call__(self, disc_generated_output, gen_output, target):
        gan_loss = self.binary_crossentropy(
            tf.ones_like(disc_generated_output), disc_generated_output
        )
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.lam * l1_loss)

        return total_gen_loss, gan_loss, l1_loss


class DiscriminatorLoss:
    def __call__(self, disc_real_output, disc_generated_output):
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        real_loss = binary_crossentropy(
            tf.ones_like(disc_real_output), disc_real_output
        )
        generated_loss = binary_crossentropy(
            tf.zeros_like(disc_generated_output), disc_generated_output
        )

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
