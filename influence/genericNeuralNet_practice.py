from __future__ import division, print_function, absolute_import, unicode_literals

import time
import os.path
import numpy as np
import tensorflow as tf
from six.moves import range
from keras import backend as K
from scipy.optimize import fmin_ncg
from tensorflow.python.ops import array_ops
from tensorflow.contrib.learn.python.learn.datasets import base

from influence.hessians import hessian_vector_product
from influence.dataset import DataSet

def variable(name, shape, initializer):
    dtype = tf.float32
    var = tf.get_variable(
        name,
        shape,
        initializer=initializer,
        dtype=dtype
    )
    return var

def variable_with_weight_decay(name,shape,stddev,wd):
    """
    similar to variable(), helper to create an initialized tf Variable
    with weight decay specified by wd, which adds L2 loss weight decay
    multiplied by this float.
    weight decay: penalise large weights in loss function to avoid overfitting.
    """
    dtype = tf.float32
    var = variable(name, shape,
                   tf.truncated_normal_initializer(stddev=stddev,dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        # add this weight decaying term to the graph collection
    return var

def normalize_vector(v):
    """
    Takes in a vector in list form, concatenates it to form a single
    vector. Normalize the vector to unit length, then returns it in list
    form together with its norm.
    """
    norm_val = np.linalg.norm(np.concatenate(v))
    # np.concatenate(v): reduce dimension of `v` to one dimension smaller than its original
    # by collapsing the first two dimensions into one.
    norm_v = [a/norm_val for a in v]
    return norm_v, norm_val



class GenericNeuralNet(object):
    """Multi-class classification neural nets."""
    def __init__(self, **kwargs):
        np.random.seed(0)
        tf.set_random_seed(0)

        self.batch_size = kwargs.pop('batch_size')
        self.data_sets = kwargs.pop('data_sets')
        self.train_dir = kwargs.pop('train_dir', 'output')
        log_dir = kwargs.pop('log_dir', 'log')
        self.model_name = kwargs.pop('model_name')
        self.num_classes = kwargs.pop('num_classes')
        self.initial_learning_rate = kwargs.pop('initial_learning_rate')
        self.decay_epochs = kwargs.pop('decay_epochs')

        self.keep_probs = kwargs.pop('keep_probs') if 'keep_probs' in kwargs else None
        self.mini_batch = kwargs.pop('mini_batch') if 'mini_batch' in kwargs else True
        self.damping = kwargs.pop('damping') if 'damping' in kwargs else 0.0

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # initialize session
        config = tf.ConfigProto() # for gpu configuration
        self.sess = tf.Session(config = config)
        K.set_session(self.sess)

        # setup input
        self.input_placeholder, self.labels_placeholder = self.placeholder_inputs()
        self.num_train_examples = self.data_sets.train.labels.shape[0]
        self.num_test_examples = self.data_sets.test.labels.shape[0]

        # setup inference and training
        if self.keep_probs is not None:
            self.keep_probs_placeholder = tf.placeholder(tf.float32, shape=(2))
            self.logits = self.inference(self.input_placeholder, self.keep_probs_placeholder)
        elif hasattr(self, 'inference_needs_labels'): # if object has attribute in string
            self.logits = self.inference(self.input_placeholder, self.labels_placeholder)
        else:
            self.logits = self.inference(self.input_placeholder)

        self.total_loss, self.loss_no_reg, self.indiv_loss_no_reg = self.loss(
            self.logits,
            self.labels_placeholder
        )




















