"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from typing import List, Tuple
import tensorflow as tf  # for deep learning tasks
# use resnet50 model for this task
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input


class RotationNetModel(tf.keras.models.Model):
    preprocess_inputs = preprocess_input

    def __init__(self,
                 image_shape: Tuple[int] = (244, 244, 3),
                 num_classes: int = 4,
                 num_hidden_units: List[int] = [512, 512],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.base_model = ResNet50V2(weights=None,
                                     include_top=False,
                                     input_shape=image_shape)

        _clf_layers = [
            tf.keras.layers.Dense(hidden) for hidden in num_hidden_units
        ]
        _clf_layers.append(tf.keras.layers.Dense(num_classes))
        self.classifier = tf.keras.models.Sequential(_clf_layers)

    def call(self, inputs, training=None, mask=None):
        x = self.preprocess_inputs(inputs)

        feature_extractor = self.base_model(inputs,training=training)

        x = self.classifier(feature_extractor,training=training)

        return x
