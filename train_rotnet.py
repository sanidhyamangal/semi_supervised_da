"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import os  # for os related ops
import argparse  # for argument parsing
import tensorflow as tf  # for deep learning ops
from models import RotationNetModel
from datapipeline.load_imageds import LoadData

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_SET = "/data/maiziezhou_lab/sanidhya/rotnet/rotated_data_train"
TEST_SET = "/data/maiziezhou_lab/sanidhya/rotnet/rotated_data_test"

train_dataset_loader = LoadData(path=TRAIN_SET,
                                image_shape=(244, 244),
                                channel=3)

val_dataset_loader = LoadData(path=TEST_SET, image_shape=(244, 244), channel=3)

train_dataset = train_dataset_loader.create_dataset(batch_size=64,
                                                    autotune=AUTOTUNE,
                                                    drop_remainder=True,
                                                    prefetch=True, cache=True)

# prepare validation dataset for the ingestion process
validation_dataset = val_dataset_loader.create_dataset(batch_size=64,
                                                       autotune=AUTOTUNE,
                                                       drop_remainder=True,
                                                       prefetch=True, cache=True)

rotnet = RotationNetModel()

rotnet.compile(
    loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
    metrics=['accuracy'])

tb_callback = tf.keras.callbacks.TensorBoard(log_dir="rotnet_101",
                                             histogram_freq=1)
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath="rotnet_models_101",
                                                   save_weights_only=True)

_tb_callback = tf.keras.callbacks.TensorBoard(log_dir="rotnet_logs",
                                              histogram_freq=1)

_model_check_points = tf.keras.callbacks.ModelCheckpoint(
    filepath="rotnet_models", save_best_only=True)

# if validation_dataset:
rotnet.fit(train_dataset,
           validation_data=validation_dataset,
           epochs=125,
           callbacks=[_tb_callback, _model_check_points])

# rotnet.fit(train_dataset, validation_data=validation_dataset,epochs=500, callbacks=[tb_callback, ckpt_callback])
