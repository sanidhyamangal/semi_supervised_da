"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from typing import List, Tuple

import tensorflow as tf  # for deep learning tasks
from tensorflow.keras.applications.resnet_v2 import \
    preprocess_input  # func to process the input to resnet

from layers import UnitNormLayer  # unit nor layer for contrastive learning


class RotationNetModel(tf.keras.models.Model):
    """
    Defining the rotation net models
    """
    def __init__(self,
                 image_shape: Tuple[int] = (244, 244, 3),
                 num_classes: int = 4,
                 num_hidden_units: int = 512,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # define the base model i.e. feature extractor
        self.base_model = tf.keras.applications.ResNet50V2(
            include_top=False, input_shape=image_shape)

        # clf layer for determining the rotation net performance
        _clf_layers = [
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_hidden_units),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_hidden_units),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes)
        ]

        # defining the classifier model based on classifier layers, stack it into a seq model.
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
        
        # select the base model as resnet50 and include the weights, images and input shape from the constructor
        self.base_model = tf.keras.applications.ResNet50V2(
            include_top=False, input_shape=image_shape, weights=weights)

        # create a hidden layer stack dynamically based on the number of hidden layers supplied
        _clf_hidden_layer_stack = []

        for hidden_states in num_hidden_units:
            _clf_hidden_layer_stack.extend([
                tf.keras.layers.Dense(hidden_states, activation="relu"),
                tf.keras.layers.Dropout(dropout)
            ])
        
        # create clf layers along with the output layer for the network
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
    """
    SuperCon Encoder model which performs the ops r = E(x)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # define the base norm layer
        self.base_model = tf.keras.applications.ResNet50V2(weights=None,
                                                           include_top=False)

        # create the embeddings along with the unit norm
        self.embedding_layers = tf.keras.models.Sequential(
            [tf.keras.layers.GlobalAveragePooling2D(),
             UnitNormLayer()])

    def call(self, inputs, training=None, mask=None):
        x = preprocess_input(inputs)

        encoded = self.base_model(x, training=training)

        embeddings = self.embedding_layers(encoded, training=training)

        return embeddings


def SupConProjector(units: int = 128):
    """Function to reutrn the projector network such that z = P(r)"""
    # create a projection layer based on the hidden layer
    projector = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(units),
         UnitNormLayer()])

    return projector
