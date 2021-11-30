#!/usr/local/bin/python3
"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import argparse
import numpy as np  # for argument parsing

import tensorflow as tf

from datapipeline.load_imageds import (  # model pipeline for loading image datasets
    LoadData, PredictionDataLoader)
from logger import logger  # for logging
from models import PAC
from trainer import BaseTrainer, UnsupervisedTrainer  # model manager for handing all the ops


def train_model(args) -> None:
    """
    Helper function for train arg subparser to train the entire network
    """
    # define data loader for the validation and trainer set
    source_dataset_loader = [
        LoadData(path=_path,
                 image_shape=(args.height, args.width),
                 channel=args.channel) for _path in args.path_to_source_dir
    ]

    target_dataset_loader = [
        LoadData(path=_path,
                 image_shape=(args.height, args.width),
                 channel=args.channel) for _path in args.path_to_target_dir
    ]

    unlabeled_dataset_loader = [
        LoadData(path=_path,
                 image_shape=(args.height, args.width),
                 channel=args.channel) for _path in args.path_to_unlabeled_dir
    ]

    logger.info(f"PRETRAINED WEIGHTS: {args.path_to_pretrained_weights}")

    # retrieve and define the model for the interconnection
    model = PAC(image_shape=(args.height, args.width, args.channel),
                num_hidden_units=[512, 512],
                num_classes=len(source_dataset_loader[0].root_labels),
                weights=args.path_to_pretrained_weights)

    DATA_DICT = {
        "source": {
            "dataloader":
            source_dataset_loader,
            "dataset":
            source_dataset_loader[0].create_dataset(batch_size=args.batch_size,
                                                    autotune=AUTOTUNE,
                                                    drop_remainder=True,
                                                    prefetch=True)
        },
        "target": {
            "dataloader":
            target_dataset_loader,
            "dataset":
            target_dataset_loader[0].create_dataset(batch_size=args.batch_size,
                                                    autotune=AUTOTUNE,
                                                    drop_remainder=True,
                                                    prefetch=True)
        },
        "unlabeled": {
            "dataloader":
            unlabeled_dataset_loader,
            "dataset":
            unlabeled_dataset_loader[0].create_dataset(
                batch_size=args.batch_size,
                autotune=AUTOTUNE,
                drop_remainder=True,
                prefetch=True,
                pertubed_images=True)
        }
    }
    # prepare the training dataset for ingesting it into the model
    # source_dataset =

    # prepare validation dataset for the ingestion process
    # validation_dataset =

    for scope in DATA_DICT:
        if len(DATA_DICT[scope]["dataloader"]) > 1:
            for i in DATA_DICT[scope]["dataloader"][1:]:
                DATA_DICT[scope]["dataset"].concatenate(
                    i.create_dataset(batch_size=args.batch_size,
                                     autotune=AUTOTUNE,
                                     drop_remainder=True,
                                     prefetch=True,
                                     pertubed_images=scope == "unlabeled"))

    trainer = BaseTrainer(model,
                          optimizer=tf.keras.optimizers.Adam(args.lr),
                          log_file_name=args.log_file_path,
                          num_classes=len(
                              source_dataset_loader[0].root_labels))
    trainer.train(args.epoch, DATA_DICT["source"]["dataset"],
                  DATA_DICT["target"]["dataset"],
                  DATA_DICT["unlabeled"]["dataset"], args.path_to_save_weights)


def train_model_unsupervised(args) -> None:
    """
    Helper function for train arg subparser to train the entire network
    """
    # define data loader for the validation and trainer set
    source_dataset_loader = [
        LoadData(path=_path,
                 image_shape=(args.height, args.width),
                 channel=args.channel) for _path in args.path_to_source_dir
    ]

    unlabeled_dataset_loader = [
        LoadData(path=_path,
                 image_shape=(args.height, args.width),
                 channel=args.channel) for _path in args.path_to_unlabeled_dir
    ]

    logger.info(f"PRETRAINED WEIGHTS: {args.path_to_pretrained_weights}")

    # retrieve and define the model for the interconnection
    model = PAC(image_shape=(args.height, args.width, args.channel),
                num_hidden_units=[512, 512],
                num_classes=len(source_dataset_loader[0].root_labels),
                weights=args.path_to_pretrained_weights)

    DATA_DICT = {
        "source": {
            "dataloader":
            source_dataset_loader,
            "dataset":
            source_dataset_loader[0].create_dataset(batch_size=args.batch_size,
                                                    autotune=AUTOTUNE,
                                                    drop_remainder=True,
                                                    prefetch=True)
        },
        "unlabeled": {
            "dataloader":
            unlabeled_dataset_loader,
            "dataset":
            unlabeled_dataset_loader[0].create_dataset(
                batch_size=args.batch_size,
                autotune=AUTOTUNE,
                drop_remainder=True,
                prefetch=True,
                pertubed_images=True)
        }
    }

    for scope in DATA_DICT:
        if len(DATA_DICT[scope]["dataloader"]) > 1:
            for i in DATA_DICT[scope]["dataloader"][1:]:
                DATA_DICT[scope]["dataset"].concatenate(
                    i.create_dataset(batch_size=args.batch_size,
                                     autotune=AUTOTUNE,
                                     drop_remainder=True,
                                     prefetch=True,
                                     pertubed_images=scope == "unlabeled"))

    trainer = UnsupervisedTrainer(model,
                                  optimizer=tf.keras.optimizers.Adam(args.lr),
                                  log_file_name=args.log_file_path,
                                  num_classes=len(
                                      source_dataset_loader[0].root_labels))

    trainer.train(args.epoch, DATA_DICT["source"]["dataset"],
                  DATA_DICT["unlabeled"]["dataset"], args.path_to_save_weights)


def evaluate(args) -> None:

    prediction_dataloader = LoadData(args.path_to_domain_dir, image_shape=[args.height, args.width])
    prediction_dataset = prediction_dataloader.create_dataset(args.batch_size, autotune=AUTOTUNE)
    
    model = PAC(image_shape=(args.height, args.width, args.channel),
                num_hidden_units=[512, 512],
                num_classes=len(prediction_dataloader.root_labels))

    model.build(input_shape=(args.height, args.width, args.channel))
    model.load_weights(args.path_to_saved_weights)


    accuracy = tf.keras.metrics.Accuracy()

    _accuracy_metric = []

    for imgs, lbls in prediction_dataset:
        _pred = model(imgs)
        _accu = accuracy(lbls, tf.argmax(_pred, axis=1))
        _accuracy_metric.append(_accu.numpy())
    

    logger.info(f"EVALUATION ACCURACY:{np.mean(_accuracy_metric)}")
    


if __name__ == "__main__":
    AUTOTUNE = tf.data.AUTOTUNE
    parser = argparse.ArgumentParser(
        description="Script to train the models for the domain adaptation")
    subparsers = parser.add_subparsers(help='semi, unsupervised')

    parser_semi = subparsers.add_parser(
        'semi',
        help='train the classification model in semi supervised fashion')
    parser_un = subparsers.add_parser(
        "unsupervised",
        help='train the classification model in semi supervised fashion')

    parser_eval = subparsers.add_parser(
        'eval', help='Evaluate the performance for the target domain')
    
    parser_semi.add_argument('--epoch',
                             type=int,
                             default=2,
                             help='number of total epoches',
                             dest="epoch")
    parser_semi.add_argument('--batch_size',
                             type=int,
                             default=32,
                             help='number of samples in one batch',
                             dest="batch_size")
    parser_semi.add_argument('--lr',
                             type=float,
                             default=1e-3,
                             help='initial learning rate for adam',
                             dest="lr")
    parser_semi.add_argument('--path_to_source_dir',
                             required=True,
                             help='path to source dataset directory',
                             dest="path_to_source_dir",
                             action="append")
    parser_semi.add_argument('--path_to_target_dir',
                             required=True,
                             help='path to target dataset directory',
                             dest="path_to_target_dir",
                             action="append")
    parser_semi.add_argument('--path_to_unlabeled_dir',
                             required=True,
                             help='path to unlabeled dataset directory',
                             dest="path_to_unlabeled_dir",
                             action="append")

    parser_semi.add_argument(
        '--path_to_pretrained_weights',
        help=
        'path to directory from where pretrained weights needs to be loaded, default to rotnet',
        dest="path_to_pretrained_weights",
        default="pretrained_models/rotnet.h5")
    parser_semi.add_argument(
        '--path_to_save_weights',
        required=True,
        help='path to directory where checkpoints needs to be saved',
        dest="path_to_save_weights",
        default="pretrained_models/rotnet.h5")
    parser_semi.add_argument(
        '--log_file_path',
        required=True,
        help='file name to save the model training loss in csv file',
        dest="log_file_path")

    parser_semi.add_argument(
        "--height",
        default=224,
        type=int,
        help="height of input images, default value is 224",
        dest="height")
    parser_semi.add_argument(
        "--width",
        default=224,
        type=int,
        help="width of input images, default value is 224",
        dest="width")
    parser_semi.add_argument(
        "--channel",
        default=3,
        type=int,
        help="channel of input images, default value is 3",
        dest="channel")

    parser_semi.set_defaults(func=train_model)

    parser_un.add_argument('--epoch',
                           type=int,
                           default=2,
                           help='number of total epoches',
                           dest="epoch")
    parser_un.add_argument('--batch_size',
                           type=int,
                           default=32,
                           help='number of samples in one batch',
                           dest="batch_size")
    parser_un.add_argument('--lr',
                           type=float,
                           default=1e-3,
                           help='initial learning rate for adam',
                           dest="lr")
    parser_un.add_argument('--path_to_source_dir',
                           required=True,
                           help='path to source dataset directory',
                           dest="path_to_source_dir",
                           action="append")
    parser_un.add_argument('--path_to_unlabeled_dir',
                           required=True,
                           help='path to unlabeled dataset directory',
                           dest="path_to_unlabeled_dir",
                           action="append")

    parser_un.add_argument(
        '--path_to_pretrained_weights',
        help=
        'path to directory from where pretrained weights needs to be loaded, default to rotnet',
        dest="path_to_pretrained_weights",
        default="pretrained_models/rotnet.h5")
    parser_un.add_argument(
        '--path_to_save_weights',
        required=True,
        help='path to directory where checkpoints needs to be saved',
        dest="path_to_save_weights",
        default="pretrained_models/rotnet.h5")
    parser_un.add_argument(
        '--log_file_path',
        required=True,
        help='file name to save the model training loss in csv file',
        dest="log_file_path")

    parser_un.add_argument("--height",
                           default=224,
                           type=int,
                           help="height of input images, default value is 224",
                           dest="height")
    parser_un.add_argument("--width",
                           default=224,
                           type=int,
                           help="width of input images, default value is 224",
                           dest="width")
    parser_un.add_argument("--channel",
                           default=3,
                           type=int,
                           help="channel of input images, default value is 3",
                           dest="channel")

    parser_un.set_defaults(func=train_model_unsupervised)


    parser_eval.add_argument('--batch_size',
                             type=int,
                             default=32,
                             help='number of samples in one batch',
                             dest="batch_size")
    
    parser_eval.add_argument('--path_to_domain_dir',
                             required=True,
                             help='path to domain directory to eval',
                             dest="path_to_domain_dir")
    parser_eval.add_argument(
        '--path_to_saved_weights',
        required=True,
        help=
        'path to directory from where pretrained weights needs to be loaded, default to rotnet',
        dest="path_to_saved_weights")

    parser_eval.add_argument(
        "--height",
        default=224,
        type=int,
        help="height of input images, default value is 224",
        dest="height")
    parser_eval.add_argument(
        "--width",
        default=224,
        type=int,
        help="width of input images, default value is 224",
        dest="width")
    parser_eval.add_argument(
        "--channel",
        default=3,
        type=int,
        help="channel of input images, default value is 3",
        dest="channel")

    parser_eval.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)
