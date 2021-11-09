"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from typing import List, Tuple
import tensorflow as tf  # for deep learning tasks
# use resnet50 model for this task
from tensorflow.keras.applications.resnet_v2 import preprocess_input


class RotationNetModel(tf.keras.models.Model):

    def __init__(self,
                 image_shape: Tuple[int] = (244, 244, 3),
                 num_classes: int = 4,
                 num_hidden_units: int = 256,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                     input_shape=image_shape)

        _clf_layers = [
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_hidden_units),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_hidden_units),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes)
        ]

        self.classifier = tf.keras.models.Sequential(_clf_layers)

    def call(self, inputs, training=None, mask=None):
        x = preprocess_input(inputs)

        feature_extractor = self.base_model(x, training=training)

        x = self.classifier(feature_extractor, training=training)
    
        return x
