"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import tensorflow as tf 

from datapipeline.load_imageds import LoadData
from models import PAC
from trainer import BaseTrainer


source_dataloader = LoadData("/data/maiziezhou_lab/sanidhya/rotnet/OfficeHomeDataset_10072016/Product", image_shape=[244,244])
target_dataloader = LoadData("/data/maiziezhou_lab/sanidhya/rotnet/semi_data/Real_World/test", image_shape=[244,244])
unlabelled_dataloader = LoadData("/data/maiziezhou_lab/sanidhya/rotnet/semi_data/Real_World/train", image_shape=[244,244])
model = PAC()
optimizer = tf.keras.optimizers.Adam(1e-3)

target_dataset = target_dataloader.create_dataset(16, autotune=tf.data.experimental.AUTOTUNE)
target_dataset = target_dataloader.create_dataset(16, autotune=tf.data.experimental.AUTOTUNE)
source_dataset = source_dataloader.create_dataset(16, autotune=tf.data.experimental.AUTOTUNE)
unlabelled_dataset = unlabelled_dataloader.create_dataset(16, autotune=tf.data.experimental.AUTOTUNE,pertubed_images=True)
trainer = BaseTrainer(model, optimizer, "logs/sample.csv")
trainer.train(2, source_dataset, target_dataset, unlabelled_dataset, "trained/pac.h5")