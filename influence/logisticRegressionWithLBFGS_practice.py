from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse 

import os.path
import time
import tensorflow as tf
import math

from tensorflow.python.ops import array_ops

from influence.hessians import hessians
from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay

class LogisticRegressionWithLBFGS(GenericNeuralNet):
    def __init__(self, input_dim, weight_decay, max_lbfgs_iter, **kwargs):
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        self.max_lbfgs_iter = max_lbfgs_iter

        super(LogisticRegressionWithLBFGS, self).__init__(**kwargs)

        self.set_params_op = self.set_params()

        self.sklearn_model = linear_model.LogisticRegression(
            C = 1/(self.num_train_examples*self.weight_decay),
            tol = 1e-8,
            fit_intercept=False,
            solver='lbfgs',
            multi_class='multinomial',
            warm_start=True,
            max_iter=max_lbfgs_iter)

        self.sklearn_model_minus_one = linear_model.LogisticRegression(
            C=1/((self.num_train_examples-1)*weight_decay),
            tol=1e-8,
            fit_intercept=False,
            solver='lbfgs',
            multi_class='multinomial',
            warm_start=True,
            max_iter=max_lbfgs_iter)

    def get_all_params(self):
        layer = 'softmax_linear'
        var_name = 'weights'
        all_params = [tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))]
        return all_params

    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32,
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder

    def inference(self, input):
        with tf.variables_scope('softmax_linear'):
            weights = variable_with_weight_decay(
                'weights',
                [self.input_di]
            )


