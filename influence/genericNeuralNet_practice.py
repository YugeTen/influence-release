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
from scipy.optimize import fmin_ncg

import os.path
import time
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K
from tensorflow.contrib.learn.python.learn.datasets import base

from influence.hessians import hessian_vector_product
from influence.dataset import DataSet


def variable(name, shape, initializer):
    dtype = tf.float32
    var = tf.get_variable(
        name,
        shape,
        initializer=initializer,
        dtype=dtype)
    return var


def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = variable(
        name,
        shape,
        initializer=tf.truncated_normal_initializer(
            stddev=stddev,
            dtype=dtype))

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def normalize_vector(v):
    """
    Takes in a vector in list form, concatenates it to form a single vector,
    normalizes it to unit length, then returns it in list form together with its norm.
    """
    norm_val = np.linalg.norm(np.concatenate(v))
    norm_v = [a / norm_val for a in v]
    return norm_v, norm_val

class GenericNeuralNet(object):
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

        if 'keep_probs' in kwargs:
            self.keep_probs = kwargs.pop('keep_probs')
        else:
            self.keep_probs = None

        if 'mini_batch' in kwargs:
            self.mini_batch = kwargs.pop('mini_batch')
        else:
            self.mini_batch = True

        if 'damping' in kwargs:
            self.damping = kwargs.pop('damping')
        else:
            self.damping = 0.0

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # Initialize session
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

        # Setup input
        self.input_placeholder, self.labels_placeholder = self.placeholder_inputs()
        self.num_train_examples = self.data_sets.train.labels.shape[0]
        self.num_test_examples = self.data_sets.test.labels.shape[0]

        # Setup inference and training
        if self.keep_probs is not None:
            self.keep_probs_placeholder = tf.placeholder(tf.float32, shape=(2))
            self.logits = self.inference(self.input_placeholder, self.keep_probs_placeholder)
        elif hasattr(self, 'inference_needs_labels'):
            self.logits = self.inference(self.input_placeholder, self.labels_placeholder)
        else:
            self.logits = self.inference(self.input_placeholder)

        self.total_loss, self.loss_no_reg, self.indiv_loss_no_reg = self.loss(
            self.logits,
            self.labels_placeholder)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.Variable(self.initial_learning_rate, name='learning_rate', trainable=False)
        self.learning_rate_placeholder = tf.placeholder(tf.float32)
        self.update_learning_rate_op = tf.assign(self.learning_rate, self.learning_rate_placeholder)

        self.train_op = self.get_train_op(self.total_loss, self.global_step, self.learning_rate)
        self.train_sgd_op = self.get_train_sgd_op(self.total_loss, self.global_step, self.learning_rate)
        self.accuracy_op = self.get_accuracy_op(self.logits, self.labels_placeholder)
        self.preds = self.predictions(self.logits)

        # Setup misc
        self.saver = tf.train.Saver()

        # Setup gradients and Hessians
        self.params = self.get_all_params()
        self.grad_total_loss_op = tf.gradients(self.total_loss, self.params)
        self.grad_loss_no_reg_op = tf.gradients(self.loss_no_reg, self.params)
        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]
        self.u_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]

        self.hessian_vector = hessian_vector_product(self.total_loss, self.params, self.v_placeholder)

        self.grad_loss_wrt_input_op = tf.gradients(self.total_loss, self.input_placeholder)

        # Because tf.gradients auto accumulates, we probably don't need the add_n (or even reduce_sum)
        self.influence_op = tf.add_n(
            [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) for a, b in
             zip(self.grad_total_loss_op, self.v_placeholder)])

        self.grad_influence_wrt_input_op = tf.gradients(self.influence_op, self.input_placeholder)

        self.checkpoint_file = os.path.join(self.train_dir, "%s-checkpoint" % self.model_name)

        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.vec_to_list = self.get_vec_to_list_fn()
        self.adversarial_loss, self.indiv_adversarial_loss = self.adversarial_loss(self.logits, self.labels_placeholder)
        if self.adversarial_loss is not None:
            self.grad_adversarial_loss_op = tf.gradients(self.adversarial_loss, self.params)

    def get_vec_to_list_fn(self):
        params_val = self.sess.run(self.params)
        self.num_params = len(np.concatenate(params_val))
        print('Total number of parameters: %s' % self.num_params)

        def vec_to_list(v):
            return_list = []
            current_position = 0
            for p in params_val:
                return_list.append(v[current_position:current_position+len(p)])
                current_position += len(p)
            assert current_position == len(v)
            return return_list

        return vec_to_list

    def reset_datasets(self):
        for data_set in self.data_sets:
            if data_set is not None:
                data_set.reset_batch()

    def fill_feed_dict_with_all_ex(self, data_set):
        feed_dict = {
            self.input_placeholder: data_set.x,
            self.labels_placeholder: data_set.labels
        }
        return feed_dict

    def fill_feed_dict_with_all_but_one_ex(self, data_set, idx_to_remove):
        num_examples = data_set.x.shape[0]
        idx = np.array([True]*num_examples, dtype=bool)
        idx[idx_to_remove] = False
        feed_dict = {
            self.input_placeholder: data_set.x[idx,:],
            self.labels_placeholder: data_set.labels[idx]
        }
        return feed_dict

    def fill_feed_dict_with_batch(self, data_set, batch_size=0):
        if batch_size is None:
            return self.fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size

        input_feed, labels_feed = data_set.next_batch(batch_size)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def fill_feed_dict_with_some_ex(self, data_set, target_indices):
        input_feed = data_set.x[target_indices, :].reshape(len(target_indices), -1)
        labels_feed = data_set.labels[target_indices].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def fill_feed_dict_with_one_ex(self, data_set, target_idx):
        input_feed = data_set.x[target_idx, :].reshape(1, -1)
        labels_feed = data_set.labels[target_idx].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def fill_feed_dict_manual(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        input_feed = X.reshape(len(Y), -1)
        labels_feed = Y.reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def minibatch_mean_eval(self, ops, data_set):
        num_examples = data_set.num_examples
        assert num_examples % self.batch_size == 0
        num_iter = int(num_examples / self.batch_size)

        self.reset_datasets()

        ret = []
        for i in range(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(data_set)
            ret_temp = self.sess.run(ops, feed_dict=feed_dict)

            if len(ret) == 0:
                for b in ret_temp:
                    if isinstance(b, list):
                        ret.append([c/float(num_iter) for c in b])
                    else:
                        ret.append([b/float(num_iter)])
            else:
                for counter, b in enumerate(ret_temp):
                    if isinstance(b, list):
                        ret[counter] = [a + (c / float(num_iter)) for (a, c) in zip(ret[counter], b)]
                    else:
                        ret[counter] += (b / float(num_iter))
        return ret

    def print_model_eval(self):
        params_val = self.sess.run(self.params)

        # when batch training
        if self.mini_batch == True:
            grad_loss_val, loss_no_reg_val, loss_val, train_acc_val = self.minibatch_mean_eval(
                [self.grad_total_loss_op,   # d(total_loss)/d(parameters)
                 self.loss_no_reg,          # yielded from function loss()
                 self.total_loss,           # yielded from function loss()
                 self.accuracy_op],         # yielded from get_accuracy_op(logits, labels_placeholder)
                self.data_sets.train)

            test_loss_val, test_acc_val = self.minibatch_mean_eval(
                [self.loss_no_reg,          # yielded from function loss()
                 self.accuracy_op],         # yielded from function get_accuracy_op(logits, labels_placeholder)
                self.data_sets.test)
        # when not batch training, no mean value needs to be taken, directly populate
        # feed_dict with all instances
        else:
            grad_loss_val, loss_no_reg_val, loss_val, train_acc_val = self.sess.run(
                [self.grad_total_loss_op,
                 self.loss_no_reg,
                 self.total_loss,
                 self.accuracy_op],
                feed_dict=self.all_train_feed_dict)

            test_loss_val, test_acc_val = self.sess.run(
                [self.loss_no_reg,
                 self.accuracy_op],
                feed_dict=self.all_test_feed_dict)

        print('Train loss (w reg) on all data: %s' % loss_val)
        print('Train loss (w/o reg) on all data: %s' % loss_no_reg_val)

        print('Test loss (w/o reg) on all data: %s' % test_loss_val)
        print('Train acc on all data:  %s' % train_acc_val)
        print('Test acc on all data:   %s' % test_acc_val)

        print('Norm of the mean of gradients: %s' % np.linalg.norm(np.concatenate(grad_loss_val)))
        print('Norm of the params: %s' % np.linalg.norm(np.concatenate(params_val)))


    def retrain(self, num_steps, feed_dict):
        for step in range(num_steps):
            self.sess.run(self.train_op, feed_dict=feed_dict)

    def update_learning_rate(self, step):
        assert self.num_train_examples % self.batch_size == 0
        num_steps_in_epoch = self.num_train_examples / self.batch_size
        epoch = step // num_steps_in_epoch

        multiplier = 1
        if epoch < self.decay_epochs[0]:
            multiplier = 1
        elif epoch < self.decay_epochs[1]:
            multiplier = 0.1
        else:
            multiplier = 0.01

        self.sess.run(
            self.update_learning_rate_op,
            feed_dict={self.learning_rate_placeholder: multiplier * self.initial_learning_rate})

    def train(self, num_steps,
              iter_to_switch_to_batch = 20000,
              iter_to_switch_to_sgd = 40000,
              save_checkpoints = True,
              verbose = True):

        if verbose: print('Training for %s steps' % num_steps)

        sess = self.sess

        for step in range(num_steps):
            self.update_learning_rate(step) # supervises the update of learning rate

            start_time = time.time()

            # mini-batch training
            if step < iter_to_switch_to_batch:
                feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train)
                _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)

            # whole batch traininig
            elif step < iter_to_switch_to_sgd:
                feed_dict = self.all_train_feed_dict
                _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict = feed_dict)

            # sgd training
            else:
                feed_dict = self.all_train_feed_dict
                _, loss_val = sess.run([self.train_sgd_op, self.total_loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            if verbose:
                if step % 1000 == 0:
                    print('Step %d: loss = %.8f (%.3f sec)' % (step, loss_val, duration))

            if (step+1) % 100000 == 0 or (step+1) == num_steps:
                if save_checkpoints: self.saver.save(sess, self.checkpoint_file, global_step=step)
                if verbose: self.print_model_eval()

    def load_checkpoint(self, iter_to_load, do_checks = True):
        checkpoint_to_load = "%s-%s" % (self.checkpoint_file, iter_to_load)
        self.saver.restore(self.sess, checkpoint_to_load)

        if do_checks:
            print('Model %s loaded. Sanity checks ---' % checkpoint_to_load)
            self.print_model_eval()

    def get_train_op(self, total_loss, global_step, learning_rate):
        """
        Return train_op
        """
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op

    def get_train_sgd_op(self, total_loss, global_step, learning_rate=0.001):
        """
        Return train_sgd_op
        """
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op



    def get_accuracy_op(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32)) / tf.shape(labels)[0]