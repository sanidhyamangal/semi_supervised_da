"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import tensorflow as tf
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

    def train_step(self, source_dataset, target_dataset, unlabelled_dataset):
        _source_loss = []
        _target_loss = []
        _cr_loss = []
        with tf.GradientTape() as tape:
            for batch_source_images, batch_source_labels in source_dataset:
                _src_preds = self.model(batch_source_images)
                _src_one_hot_labels = tf.one_hot(batch_source_labels,
                                                 depth=self.num_classes)
                _src_loss = compute_h(_src_one_hot_labels, _src_preds)
                _source_loss.append(_src_loss)

            for batch_target_images, batch_target_labels in target_dataset:
                _tgt_preds = self.model(batch_target_images)
                _tgt_one_hot_labels = tf.one_hot(batch_target_labels,
                                                 depth=self.num_classes)
                _tgt_loss = compute_h(_tgt_one_hot_labels, _tgt_preds)
                _target_loss.append(_tgt_loss)

            for batch_unlabelled_images in unlabelled_dataset:
                batch_perturbed_images = GeneratePertuberations(
                    batch_unlabelled_images)

                px = self.model(batch_unlabelled_images)
                qx = self.model(batch_perturbed_images)

                _c_loss = compute_cr(px, qx, 0.9)
                _cr_loss.append(_c_loss)

            loss = tf.reduce_mean(_source_loss) + tf.reduce_mean(
                _target_loss) + tf.reduce_mean(_cr_loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return loss

    def train(self,
              epochs: int,
              source_dataset,
              target_dataset,
              unlabelled_dataset,
              weights_path="str"):
        for i in epochs:
            # call training step
            loss = self.train_step(source_dataset, target_dataset,
                                   unlabelled_dataset)
            logger.info(f"Epoch: {i},Loss:{loss}")
            self.write_logs_csv(loss)

            if loss < self.base_loss:
                self.save_weights(weights_path)
                self.base_loss = loss
                logger.info(f"Saving Weights at Epoch :{i} - Loss:{loss}")

    def save_weights(self, path: str) -> None:
        create_folders_if_not_exists(path)  # save path
        self.model.save_weights(path)  # save weights
