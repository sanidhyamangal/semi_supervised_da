"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import numpy as np
import tensorflow as tf

from contrastive_loss import \
    max_margin_contrastive_loss  # for marginal contrastive loss
from datapipeline.transforms import GeneratePertuberations  # for deep learning
from logger import logger
from losses import compute_cr, compute_h  # for loss related ops
from models import SupConProjector, SupervisedContrastiveEncoder
from utils import create_folders_if_not_exists


class BaseTrainer:
    """
    Define base trainer to abstract the training ops for all the models.
    """
    def __init__(self,
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Adam,
                 log_file_name: str = "logs.csv",
                 num_classes: int = 65) -> None:
        
        # define all the models and base models for the training ops
        self.model = model
        self.optimizer = optimizer
        self.log_file_writer = log_file_name
        self.num_classes = num_classes
        self.base_loss = float("inf")

        # call for the create log file writer logs
        create_folders_if_not_exists(self.log_file_writer)

    def write_logs_csv(self, loss) -> None:
        """
        A small subroutine for writing the csv log files
        """
        with open(self.log_file_writer, "a+") as fp:
            fp.write(f"{loss}\n")


    def train_step(self, source_batch, target_batch, unlabeled_batch):
        """
        A function to handle the minibatch step
        """
        loss = 0.0 # define the base loss

        # compute the loss on the grad tape for computing the grads and later optimizing it.
        with tf.GradientTape() as tape:

            # compute the source_btach loss
            if source_batch.has_value():
                # get the labels and images from the dataset
                imgs, labels = source_batch.get_value()

                # compute the predictions
                pred = self.model(imgs)

                # compute the cross entropy loss with ground truth as px and qx as predictions
                loss += tf.reduce_mean(
                    compute_h(tf.one_hot(labels, depth=self.num_classes),
                              pred))

            # apply same steps as source minibatch
            if target_batch.has_value():
                imgs, labels = target_batch.get_value()
                pred = self.model(imgs)
                loss += tf.reduce_mean(
                    compute_h(tf.one_hot(labels, depth=self.num_classes),
                              pred))

            # compute the vals from the unlabeled datset
            imgs, pertubed_imgs = unlabeled_batch.get_value()

            # compute px and qx for finding the consistency loss
            px = self.model(imgs)
            qx = self.model(pertubed_imgs)

            # compute the final loss
            loss += compute_cr(px, qx, 0.9)

        # compute the grads based on the loss for all the trainiable variables
        grads = tape.gradient(loss, self.model.trainable_variables)

        # apply the grads via adam optimizer
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        # return the loss for storing it.
        return loss

    def train(self,
              epochs: int,
              source_dataset,
              target_dataset,
              unlabeled_dataset,
              weights_path="str"):

        # rub routine for training the ops
        for i in range(epochs):
            # call training step

            # store the epoch loss for logging
            epoch_loss = []

            # create an dataset iterator for performing the training ops
            source_iterator = iter(source_dataset)
            target_iterator = iter(target_dataset)
            unlabeled_iterator = iter(unlabeled_dataset)

            # run while source batch has value
            while True:
                # get the mini batch val for all the three datasets
                source_batch = source_iterator.get_next_as_optional()
                target_batch = target_iterator.get_next_as_optional()
                unlabeled_batch = unlabeled_iterator.get_next_as_optional()

                # breaking condition if no more instances left in source batch
                if not source_batch.has_value():
                    break
                
                # if no more instances in unlabeled batch then we reinit the iterator
                if not unlabeled_batch.has_value():
                    unlabeled_iterator = iter(unlabeled_dataset)
                    unlabeled_batch = unlabeled_iterator.get_next_as_optional()

                # if no more vals from target batch then re init the iterator
                if not target_batch.has_value():
                    target_iterator = iter(target_dataset)
                    target_batch = target_iterator.get_next_as_optional()

                # comput the loss and apply the gradient step ops
                loss = self.train_step(source_batch, target_batch,
                                       unlabeled_batch)

                # log the batch loss
                logger.info(f"Batch Loss: {loss}")

                # buffer the epoch loss
                epoch_loss.append(loss)

                # append the batch loss to the trainer
                self.write_logs_csv(loss)

            # compute the epoch loss by taking mean of all the batch loss
            ep_loss = np.mean(epoch_loss)
            logger.info(f"Epoch: {i},Loss:{ep_loss}")

            # save model if the performance of the ep loss is less than base loss
            if ep_loss < self.base_loss:
                self.save_weights(weights_path)
                self.base_loss = ep_loss
                logger.info(f"Saving Weights at Epoch :{i} - Loss:{loss}")

        logger.info("Training Finished !!!!")

    def save_weights(self, path: str) -> None:
        "Method to save the weights"
        create_folders_if_not_exists(path)  # save path
        self.model.save_weights(path)  # save weights


class UnsupervisedTrainer(BaseTrainer):
    """
    Unsupervised trainer derived from the base trainer
    The only difference is that the no labeled examples are used in this from the target domain
    """

    def train_step(self, source_batch, unlabeled_batch):
        loss = 0
        with tf.GradientTape() as tape:
            if source_batch.has_value():
                imgs, labels = source_batch.get_value()
                pred = self.model(imgs)
                loss += tf.reduce_mean(
                    compute_h(tf.one_hot(labels, depth=self.num_classes),
                              pred))

            imgs, pertubed_imgs = unlabeled_batch.get_value()

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
              unlabeled_dataset,
              weights_path="str"):

        # ops for the training
        # same as base
        for i in range(epochs):
            # call training step
            epoch_loss = []
            source_iterator = iter(source_dataset)
            unlabeled_iterator = iter(unlabeled_dataset)

            while True:
                source_batch = source_iterator.get_next_as_optional()
                unlabeled_batch = unlabeled_iterator.get_next_as_optional()

                if not source_batch.has_value():
                    break

                if not unlabeled_batch.has_value():
                    unlabeled_iterator = iter(unlabeled_dataset)
                    unlabeled_batch = unlabeled_iterator.get_next_as_optional()

                loss = self.train_step(source_batch, unlabeled_batch)
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


class SuperConTrainer(BaseTrainer):
    """
    Supervised Contrastive learning trainer to perform the trainer ops
    """
    def __init__(self,
                 encoder_model: SupervisedContrastiveEncoder,
                 projector_model: tf.keras.models.Model,
                 optimizer: tf.keras.optimizers.Adam,
                 log_file_name: str = "logs.csv") -> None:

        # define all the training ops
        self.encoder_model = encoder_model
        self.projector_model = projector_model
        self.optimizer = optimizer
        self.log_file_writer = log_file_name
        self.base_loss = float("inf")

        # call for the create log file writer logs
        create_folders_if_not_exists(self.log_file_writer)

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:

            encoder = self.encoder_model(images, training=True)
            projector = self.projector_model(encoder, training=True)

            # compute the max margin contrastive loss for training ops
            loss = max_margin_contrastive_loss(projector,
                                               labels,
                                               metric="cosine")

        # computing the gradient tape
        grads = tape.gradient(
            loss, self.encoder_model.trainable_variables +
            self.projector_model.trainable_variables)


        self.optimizer.apply_gradients(
            zip(
                grads, self.encoder_model.trainable_variables +
                self.projector_model.trainable_variables))

        return loss

    def train(self, train_steps: int, dataset, weights_path: str):

        for epoch in range(train_steps):
            epoch_loss_avg = []

            for images, labels in dataset:
                loss = self.train_step(images, labels)
                epoch_loss_avg.append(loss)
                logger.info(f"Batch Loss: {loss}")
                self.write_logs_csv(loss)

            _ep_loss = np.mean(epoch_loss_avg)
            logger.info(f"Epoch:{epoch}, Loss :{_ep_loss}")

            if _ep_loss < self.base_loss:
                self.save_weights(weights_path)
                self.base_loss = _ep_loss
                logger.info(
                    f"Saving Weights at Epoch :{epoch} - Loss:{_ep_loss}")

    def save_weights(self, path: str) -> None:
        # save weights of the base model for using it in PAC
        create_folders_if_not_exists(path)
        self.encoder_model.base_model.save_weights(path)
