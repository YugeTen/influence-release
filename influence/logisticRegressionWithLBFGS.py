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

        C = 1.0 / (self.num_train_examples * self.weight_decay)
        self.sklearn_model = linear_model.LogisticRegression(
            C=C,
            tol=1e-8,
            fit_intercept=False,
            solver='lbfgs',
            multi_class='multinomial',
            warm_start=True, #True
            max_iter=max_lbfgs_iter)

        C_minus_one = 1.0 / ((self.num_train_examples - 1) * self.weight_decay)
        self.sklearn_model_minus_one = linear_model.LogisticRegression(
            C=C_minus_one,
            tol=1e-8,
            fit_intercept=False,
            solver='lbfgs',
            multi_class='multinomial',
            warm_start=True, #True
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
        """
        initialise the softmax_linear layer variables with weight decay using
        a truncated normal distribution, weights is a tensor
        """
        with tf.variable_scope('softmax_linear'):
            weights = variable_with_weight_decay(
                name='weights',
                shape=[self.input_dim * self.num_classes],
                stddev=1.0 / math.sqrt(float(self.input_dim)),
                wd=self.weight_decay)            
            logits = tf.matmul(input, tf.reshape(weights, [self.input_dim, self.num_classes]))

        self.weights = weights

        return logits


    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds


    def set_params(self):
        self.W_placeholder = tf.placeholder(
            tf.float32,
            shape=[self.input_dim * self.num_classes],
            name='W_placeholder')
        set_weights = tf.assign(self.weights, self.W_placeholder, validate_shape=True)
        return [set_weights]



    def retrain(self, num_steps, feed_dict):
    
        self.train_with_LBFGS(
            feed_dict=feed_dict,
            save_checkpoints=False, 
            verbose=False)
        


    def train(self, num_steps=None, 
              iter_to_switch_to_batch=None, 
              iter_to_switch_to_sgd=None,
              save_checkpoints=True, verbose=True):

        self.train_with_LBFGS(
            feed_dict=self.all_train_feed_dict,
            save_checkpoints=save_checkpoints, 
            verbose=verbose)


    def train_with_LBFGS(self, feed_dict, save_checkpoints=True, verbose=True):
        # More sanity checks to see if predictions are the same?        

        X_train = feed_dict[self.input_placeholder]
        Y_train = feed_dict[self.labels_placeholder]
        num_train_examples = len(Y_train)
        assert len(Y_train.shape) == 1
        assert X_train.shape[0] == Y_train.shape[0]

        if num_train_examples == self.num_train_examples:
            if verbose: print('Using normal model')
            model = self.sklearn_model
        elif num_train_examples == self.num_train_examples - 1:
            if verbose: print('Using model minus one')
            model = self.sklearn_model_minus_one
        else:
            raise ValueError("feed_dict has incorrect number of training examples")

        model.fit(X_train, Y_train)
        # sklearn returns coefficients in shape num_classes x num_features
        # whereas our weights are defined as num_features x num_classes
        # so we have to tranpose them first.
        W = np.reshape(model.coef_.T, -1)

        params_feed_dict = {}
        params_feed_dict[self.W_placeholder] = W
        self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
        if save_checkpoints: self.saver.save(self.sess, self.checkpoint_file, global_step=0)

        if verbose:
            print('LBFGS training took %s iter.' % model.n_iter_)
            print('After training with LBFGS: ')
            self.print_model_eval()





