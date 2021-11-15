"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import tensorflow as tf  # for computing the loss
import numpy as np  # for matrix maths


class PACLoss:
    def __init__(self, tau: float = 0.2, depth: int = 65):
        self.tau = tau
        self.depth = depth

    def compute_h(self, px, qx):
        _log_softmax = tf.math.log(qx)

        H = tf.reduce_sum(tf.multiply(-px, _log_softmax), axis=1)

        return H

    def compute_cr(self, px, qx):
        _cr_list = []
        _H_px_qx = self.compute_h(px, qx)
        for _p, _h in zip(px, _H_px_qx):
            _mask = _p[tf.greater_equal(_p, self.tau)]
            if _mask.shape[0] == 0:
                _max_px = 0
            else:
                _max_px = np.max(_mask)
            _cr_list.append(_max_px * _h)

        L_cr = tf.reduce_mean(_cr_list)

        return L_cr

    def __call__(self, y_true, px, qx):
        # onehot encoded y_true
        # import pdb;pdb.set_trace()
        one_hot_y_true = tf.one_hot(y_true, depth=self.depth)

        _H_normal = tf.reduce_mean(self.compute_h(one_hot_y_true, px))
        _H_perturbed = tf.reduce_mean(self.compute_h(one_hot_y_true, qx))
        _L_cr = self.compute_cr(px, qx)

        loss = _H_normal + _H_perturbed + _L_cr

        return loss
