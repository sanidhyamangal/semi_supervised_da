"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import mean
from datapipeline.transforms import GeneratePertuberations  # for deep learning

from losses import compute_cr, compute_h  # for loss related ops
from utils import create_folders_if_not_exists
from logger import logger


class BaseTrainer:
    def __init__(self,
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Adam,
                 log_file_name: str = "logs.csv",
                 num_classes: int = 65) -> None:
        self.model = model
        self.optimizer = optimizer
        self.log_file_writer = log_file_name
        self.num_classes = num_classes
        self.base_loss = float("inf")

        # call for the create log file writer logs
        create_folders_if_not_exists(self.log_file_writer)

    def write_logs_csv(self, loss) -> None:
        with open(self.log_file_writer, "a+") as fp:
            fp.write(f"{loss}\n")

    def train_step(self, source_batch, target_batch, unlabeled_batch):
        loss = 0
        with tf.GradientTape() as tape:
            if source_batch.has_value():
                imgs, labels = source_batch.get_value()
                pred = self.model(imgs)
                loss += tf.reduce_mean(
                    compute_h(tf.one_hot(labels, depth=self.num_classes),
                              pred))

            if target_batch.has_value():
                imgs, labels = target_batch.get_value()
                pred = self.model(imgs)
                loss += tf.reduce_mean(
                    compute_h(tf.one_hot(labels, depth=self.num_classes),
                              pred))

            imgs = unlabeled_batch.get_value()
            pertubed_imgs = GeneratePertuberations(imgs)

            px = self.model(imgs)
            qx = self.model(pertubed_imgs)

            loss += compute_cr(px, qx, 0.9)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return loss

    def train(self,
              epochs: int,
              source_dataset,
              target_dataset,
              unlabeled_dataset,
              weights_path="str"):

        for i in range(epochs):
            # call training step
            epoch_loss = []
            source_iterator = iter(source_dataset)
            target_iterator = iter(target_dataset)
            unlabeled_iterator = iter(unlabeled_dataset)

            while True:
                source_batch = source_iterator.get_next_as_optional()
                target_batch = target_iterator.get_next_as_optional()
                unlabeled_batch = unlabeled_iterator.get_next_as_optional()

                if not unlabeled_batch.has_value():
                    break

                if not target_batch.has_value():
                    target_iterator = iter(target_dataset)
                    target_batch = target_iterator.get_next_as_optional()

                loss = self.train_step(source_batch, target_batch,
                                       unlabeled_batch)
                logger.info(f"Batch Loss: {loss}")
                epoch_loss.append(loss)
                self.write_logs_csv(loss)

            ep_loss = np.mean(epoch_loss)
            logger.info(f"Epoch: {i},Loss:{ep_loss}")

            if loss < self.base_loss:
                self.save_weights(weights_path)
                self.base_loss = loss
                logger.info(f"Saving Weights at Epoch :{i} - Loss:{loss}")
        
        logger.info("Training Finished !!!!")

    def save_weights(self, path: str) -> None:
        create_folders_if_not_exists(path)  # save path
        self.model.save_weights(path)  # save weights
