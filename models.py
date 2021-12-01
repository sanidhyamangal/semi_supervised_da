"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from typing import List, Tuple
import tensorflow as tf  # for deep learning tasks
# use resnet50 model for this task
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from layers import UnitNormLayer # unit nor layer for contrastive learning

class RotationNetModel(tf.keras.models.Model):
    def __init__(self,
                 image_shape: Tuple[int] = (244, 244, 3),
                 num_classes: int = 4,
                 num_hidden_units: int = 256,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.base_model = tf.keras.applications.ResNet50V2(
            include_top=False, input_shape=image_shape)

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


class PAC(tf.keras.models.Model):
    """
    Model for performing PAC
    """
    def __init__(self,
                 image_shape: Tuple[int] = (244, 244, 3),
                 num_classes: int = 65,
                 num_hidden_units: List[int] = [256, 256],
                 dropout: float = 0.3,
                 weights: str = "imagenet",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.base_model = tf.keras.applications.ResNet50V2(
            include_top=False, input_shape=image_shape, weights=weights)

        _clf_hidden_layer_stack = []

        for hidden_states in num_hidden_units:
            _clf_hidden_layer_stack.extend([
                tf.keras.layers.Dense(hidden_states, activation="relu"),
                tf.keras.layers.Dropout(dropout)
            ])

        _clf_layers = [
            tf.keras.layers.GlobalAveragePooling2D(), *_clf_hidden_layer_stack,
            tf.keras.layers.Dense(num_classes, activation="softmax")
        ]

        self.classifier = tf.keras.models.Sequential(_clf_layers)

    def call(self, inputs, training=None, mask=None):
        x = preprocess_input(inputs)

        feature_extractor = self.base_model(x, training=training)

        x = self.classifier(feature_extractor, training=training)

        return x

class SupervisedContrastiveEncoder(tf.keras.models.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_model = tf.keras.applications.ResNet50V2(weights=None, include_top=False)
        self.embedding_layers = tf.keras.models.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            UnitNormLayer()
        ])


    def call(self, inputs, training=None, mask=None):
        x = preprocess_input(inputs)

        encoded = self.base_model(x, training=training)

        embeddings = self.embedding_layers(encoded, training=training)

        return embeddings


def SupConProjector(units:int=128):
    projector = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units),
        UnitNormLayer()
    ])

    return projector
