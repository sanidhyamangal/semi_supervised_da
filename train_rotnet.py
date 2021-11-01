"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import os # for os related ops
import argparse # for argument parsing
import tensorflow as tf # for deep learning ops
from models import RotationNetModel


TRAIN_SET = "/data/maiziezhou_lab/sanidhya/rotnet/rotated_data_train"
TEST_SET = "/data/maiziezhou_lab/sanidhya/rotnet/rotated_data_test"

image_data = tf.keras.preprocessing.image.ImageDataGenerator()
train_dataset = image_data.flow_from_directory(TRAIN_SET, target_size=(244,244), batch_size=64)
test_dataset = image_data.flow_from_directory(TEST_SET, target_size=(244,244),batch_size=64)


rotnet = RotationNetModel()

rotnet.compile(loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)],metrics=["accuracy"])

tb_callback = tf.keras.callbacks.TensorBoard(log_dir="rotnet",histogram_freq=1)
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath="rotnet_models", save_weights_only=True)

rotnet.fit(train_dataset, validation_data=test_dataset,epochs=500, callbacks=[tb_callback, ckpt_callback])
