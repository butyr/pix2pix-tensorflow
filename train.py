import datetime
import os
import pathlib

import tensorflow as tf

from src.container import ExperimentLogging, Model
from src.dataloader import Dataloader
from src.loss import DiscriminatorLoss, GeneratorLoss
from src.models import PatchGAN, UNet
from src.trainer import Trainer


def main():
    # The facade training set consist of 400 images
    BUFFER_SIZE = 400
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    BATCH_SIZE = 1
    # Each image is 256x256 in size
    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    dataset_name = "facades"

    _URL = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz"

    path_to_zip = tf.keras.utils.get_file(
        fname=f"{dataset_name}.tar.gz", origin=_URL, extract=True
    )

    path_to_zip = pathlib.Path(path_to_zip)
    PATH = path_to_zip.parent / dataset_name

    log_dir = "logs/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    trainer = Trainer(
        generator=Model(
            network=UNet(num_channels=3),
            optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            loss=GeneratorLoss(lam=100),
        ),
        discriminator=Model(
            network=PatchGAN(),
            optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
            loss=DiscriminatorLoss(),
        ),
        logging=ExperimentLogging(
            summary_writer=summary_writer,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=checkpoint_prefix,
        ),
    )
    dataloader = Dataloader(
        PATH,
        BUFFER_SIZE,
        BATCH_SIZE,
        IMG_WIDTH,
        IMG_HEIGHT,
    )

    train_dataset, test_dataset = dataloader.build_train(), dataloader.build_test()
    trainer.fit(train_dataset, test_dataset, steps=40000)


if __name__ == "__main__":
    main()
