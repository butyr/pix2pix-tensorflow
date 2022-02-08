from dataclasses import dataclass
from typing import Callable

import tensorflow as tf


@dataclass
class Model:
    network: tf.keras.Model
    optimizer: tf.keras.optimizers.Optimizer
    loss: Callable


@dataclass
class ExperimentLogging:
    summary_writer: tf.summary.SummaryWriter
    checkpoint_dir: str
    checkpoint_prefix: str
