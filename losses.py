"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import tensorflow as tf  # for computing the loss
import numpy as np  # for matrix maths


def compute_h(px, qx):
    """Function to compute cross entropy for the predictions"""
    _log_softmax = tf.math.log(qx + 1e-8)

    H = tf.reduce_sum(tf.multiply(-px, _log_softmax), axis=1)

    return H


def compute_cr(px, qx, tau):
    """Function to compute the consistency regularization loss for the """

    # compute the cr function
    _cr_list = []
    _H_px_qx = compute_h(px, qx)

    # iterate in entire row
    for _p, _h in zip(px, _H_px_qx):

        # compute the mask for the indicator function
        _mask = _p[tf.greater_equal(_p, tau)]
        if _mask.shape[0] == 0:
            _max_px = 0
        else:
            _max_px = np.max(_mask)

        # append the cr loss in the max_px
        _cr_list.append(_max_px * _h)

    # compute the final L_cr.
    L_cr = tf.reduce_mean(_cr_list)

    return L_cr


class PACLoss:
    """A class based loss for computing the entire PAC loss"""
    def __init__(self, tau: float = 0.2, depth: int = 65):
        self.tau = tau
        self.depth = depth

    def compute_h(self, px, qx):
        return compute_h(px, qx)

    def compute_cr(self, px, qx):
        return compute_cr(px, qx, self.tau)

    def __call__(self, y_true, ms, mt, px, qx):
        # onehot encoded y_true
        one_hot_y_true = tf.one_hot(y_true, depth=self.depth)

        _H_source = tf.reduce_mean(self.compute_h(one_hot_y_true, ms))
        _H_target = tf.reduce_mean(self.compute_h(one_hot_y_true, mt))
        _L_cr = self.compute_cr(px, qx)

        loss = _H_source + _H_target + _L_cr

        return loss
