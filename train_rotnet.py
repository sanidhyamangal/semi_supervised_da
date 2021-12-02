"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import os  # for os related ops
import argparse  # for argument parsing
import tensorflow as tf  # for deep learning ops
from models import RotationNetModel
from datapipeline.load_imageds import LoadData
from utils import create_folders_if_not_exists

def train_model(args):
    TRAIN_SET = args.path_to_train_dir
    TEST_SET = args.path_to_test_dir

    train_dataset_loader = LoadData(path=TRAIN_SET,
                                    image_shape=(args.height, args.width), channel=args.channel)

    val_dataset_loader = LoadData(path=TEST_SET, image_shape=(args.height, args.width), channel=args.channel)

    train_dataset = train_dataset_loader.create_dataset(batch_size=32,
                                                        autotune=AUTOTUNE,
                                                        drop_remainder=True,
                                                        prefetch=True)

    # prepare validation dataset for the ingestion process
    validation_dataset = val_dataset_loader.create_dataset(batch_size=32,
                                                        autotune=AUTOTUNE,
                                                        drop_remainder=True,
                                                        prefetch=True)

    rotnet = RotationNetModel(image_shape=(args.height, args.width, args.channel))

    rotnet.compile(
        loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
        metrics=['accuracy'])

    _tb_callback = tf.keras.callbacks.TensorBoard(log_dir=args.path_to_tb,
                                                histogram_freq=1)

    _model_check_points = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.path_to_checkpoint, save_best_only=True)

    # if validation_dataset:
    rotnet.fit(train_dataset,
            validation_data=validation_dataset,
            epochs=args.epoch,
            callbacks=[_tb_callback, _model_check_points])
    
    rotnet.load_weights(args.path_to_checkpoint)
    create_folders_if_not_exists(args.path_to_save_weights)
    rotnet.base_model.save_weights(args.path_to_save_weights)

# rotnet.fit(train_dataset, validation_data=validation_dataset,epochs=500, callbacks=[tb_callback, ckpt_callback])

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
    parser.add_argument('--path_to_train_dir',
                             required=True,
                             help='path to dataset directory',
                             dest="path_to_train_dir")
    parser.add_argument('--path_to_test_dir',
                             required=True,
                             help='path to dataset directory',
                             dest="path_to_test_dir")

    parser.add_argument(
        '--path_to_checkpoint',
        required=True,
        help='path to directory where checkpoints needs to be saved',
        dest="path_to_checkpoint",
        default="rotnet_training_ckpt")
    
    parser.add_argument(
        '--path_to_save_weights',
        required=True,
        help='path to directory where checkpoints needs to be saved',
        dest="path_to_save_weights",
        default="pretrained_models/rotnet.h5")

    parser.add_argument(
        '--path_to_tb',
        required=True,
        help='file name to save the model performance as tensorboard log files',
        dest="path_to_tb")

    parser.add_argument(
        "--height",
        default=128,
        type=int,
        help="height of input images, default value is 128",
        dest="height")
    parser.add_argument(
        "--width",
        default=128,
        type=int,
        help="width of input images, default value is 128",
        dest="width")
    parser.add_argument(
        "--channel",
        default=3,
        type=int,
        help="channel of input images, default value is 3",
        dest="channel")

    parser.set_defaults(func=train_model)

    args = parser.parse_args()
    args.func(args)
