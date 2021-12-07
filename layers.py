"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import tensorflow as tf  # for deep learning


class UnitNormLayer(tf.keras.layers.Layer):
    """
    A Unit Norm layer to project all the vectors on a unit norm.
    """
    def __init__(self):
        super(UnitNormLayer, self).__init__()

    def call(self, input_tensor):
        norm = tf.norm(input_tensor, axis=1)
        return input_tensor / tf.reshape(norm, [-1, 1])
