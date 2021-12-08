#!/usr/local/bin/python3
"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import argparse

import numpy as np  # for argument parsing
import tensorflow as tf

from datapipeline.load_imageds import \
    LoadSuperConData  # model pipeline for loading image datasets
from logger import logger  # for logging
from models import SupConProjector, SupervisedContrastiveEncoder
from trainer import SuperConTrainer


def train_model(args) -> None:
    """
    Helper function for train arg subparser to train the entire network
    """
    # load the supercon dataloader
    supercon_dataloader = LoadSuperConData(args.path_to_data_dir,
                                           image_shape=(args.height,
                                                        args.width),
                                           channel=args.channel)

    # cretae supcon dataset
    supercon_dataset = supercon_dataloader.create_dataset(args.batch_size,
                                                          autotune=AUTOTUNE)
    encoder_model = SupervisedContrastiveEncoder()
    projector_model = SupConProjector()

    # define optimizer and trainer
    decay_steps = 1000
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=0.001, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.Adam(lr_decayed_fn)

    trainer = SuperConTrainer(encoder_model,
                              projector_model,
                              optimizer=optimizer,
                              log_file_name=args.log_file_path)

    # train the model
    trainer.train(args.epoch, supercon_dataset, args.path_to_save_weights)


if __name__ == "__main__":
    AUTOTUNE = tf.data.AUTOTUNE
    parser = argparse.ArgumentParser(
        description="Script to train the models for the domain adaptation")

    parser.add_argument('--epoch',
                        type=int,
                        default=2,
                        help='number of total epoches',
                        dest="epoch")
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='number of samples in one batch',
                        dest="batch_size")
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='initial learning rate for adam',
                        dest="lr")
    parser.add_argument('--path_to_data_dir',
                        required=True,
                        help='path to dataset directory',
                        dest="path_to_data_dir")

    parser.add_argument(
        '--path_to_save_weights',
        required=True,
        help='path to directory where checkpoints needs to be saved',
        dest="path_to_save_weights",
        default="pretrained_models/supercon.h5")
    parser.add_argument(
        '--log_file_path',
        required=True,
        help='file name to save the model training loss in csv file',
        dest="log_file_path")

    parser.add_argument("--height",
                        default=128,
                        type=int,
                        help="height of input images, default value is 128",
                        dest="height")
    parser.add_argument("--width",
                        default=128,
                        type=int,
                        help="width of input images, default value is 128",
                        dest="width")
    parser.add_argument("--channel",
                        default=3,
                        type=int,
                        help="channel of input images, default value is 3",
                        dest="channel")

    parser.set_defaults(func=train_model)

    args = parser.parse_args()
    args.func(args)
