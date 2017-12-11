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
    norm_v = [a/norm_val for a in v]
    return norm_v, norm_val


class GenericNeuralNet(object):
    """
    Multi-class classification.
    """

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

        if 'keep_probs' in kwargs: self.keep_probs = kwargs.pop('keep_probs')
        else: self.keep_probs = None
        
        if 'mini_batch' in kwargs: self.mini_batch = kwargs.pop('mini_batch')        
        else: self.mini_batch = True
        
        if 'damping' in kwargs: self.damping = kwargs.pop('damping')
        else: self.damping = 0.0
        
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
            [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) for a, b in zip(self.grad_total_loss_op, self.v_placeholder)])

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
        # 2 functionality:
        #   (1) Prints total number of parameters used in model
        #   (2) return vec_to_list function, which turns input v to a list
        params_val = self.sess.run(self.params)
        self.num_params = len(np.concatenate(params_val))        
        print('Total number of parameters: %s' % self.num_params)
        print('Total number of train examples: %s' % self.num_train_examples)


        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p in params_val:
                return_list.append(v[cur_pos : cur_pos+len(p)])
                cur_pos += len(p)

            assert cur_pos == len(v)
            return return_list

        return vec_to_list


    def reset_datasets(self):
        # for each test/validation/train split in self.data_sets, run reset_batch()
        # defined in DataSet class if the split is not empty
        for data_set in self.data_sets:
            if data_set is not None:
                data_set.reset_batch()


    def fill_feed_dict_with_all_ex(self, data_set):
        # with split data_set from data_sets, populate feed_dict and return
        feed_dict = {
            self.input_placeholder: data_set.x,
            self.labels_placeholder: data_set.labels
        }
        return feed_dict


    def fill_feed_dict_with_all_but_one_ex(self, data_set, idx_to_remove):
        # same as fill_feed_dict_with_all_ex, but ignore index specified
        # by idx_to_remove
        num_examples = data_set.x.shape[0]
        idx = np.array([True] * num_examples, dtype=bool)
        idx[idx_to_remove] = False
        feed_dict = {
            self.input_placeholder: data_set.x[idx, :],
            self.labels_placeholder: data_set.labels[idx]
        }
        return feed_dict


    def fill_feed_dict_with_batch(self, data_set, batch_size=0):
        # same as fill_feed_dict_with_all_ex, but fill only batch_size number
        # of samples
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
        # same as fill_feed_dict_with_all_ex, but fill only indices specified by
        # target_indices
        input_feed = data_set.x[target_indices, :].reshape(len(target_indices), -1)
        labels_feed = data_set.labels[target_indices].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict


    def fill_feed_dict_with_one_ex(self, data_set, target_idx):
        # same as fill_feed_dict_with_all_ex, but fill only index specified by
        # target_idx
        input_feed = data_set.x[target_idx, :].reshape(1, -1)
        labels_feed = data_set.labels[target_idx].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict


    def fill_feed_dict_manual(self, X, Y):
        # same as fill_feed_dict_with_all_ex, but fill only with one instance
        # of (X,Y)
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
        # helper function for print_model_eval, with the split specified by
        # data_set and operations specified by ops, evaluate the average of each
        # operation output for each minibatch

        # used for finding the mean of each mini batch of accuracy/loss/derivative
        # of loss to parameters, called by function print_model_eval
        num_examples = data_set.num_examples
        assert num_examples % self.batch_size == 0
        num_iter = int(num_examples / self.batch_size)

        self.reset_datasets()

        ret = []
        for i in range(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(data_set)
            ret_temp = self.sess.run(ops, feed_dict=feed_dict)
            
            if len(ret)==0:
                for b in ret_temp:
                    if isinstance(b, list):
                        ret.append([c / float(num_iter) for c in b])
                    else:
                        ret.append([b / float(num_iter)])
            else:
                for counter, b in enumerate(ret_temp):
                    if isinstance(b, list):
                        ret[counter] = [a + (c / float(num_iter)) for (a, c) in zip(ret[counter], b)]
                    else:
                        ret[counter] += (b / float(num_iter))
            
        return ret


    def print_model_eval(self):
        # with the aid of minibatch_mean_eval, print out:
        #   mean training loss with/without regression
        #   mean testing loss without regression
        #   mean training and testing accuracy
        #   norm of the mean of total loss to parameter gradient
        #   norm of all parameters
        params_val = self.sess.run(self.params)

        if self.mini_batch == True:
            grad_loss_val, loss_no_reg_val, loss_val, train_acc_val = self.minibatch_mean_eval(
                [self.grad_total_loss_op, self.loss_no_reg, self.total_loss, self.accuracy_op],
                self.data_sets.train)
            
            test_loss_val, test_acc_val = self.minibatch_mean_eval(
                [self.loss_no_reg, self.accuracy_op],
                self.data_sets.test)

        else:
            grad_loss_val, loss_no_reg_val, loss_val, train_acc_val = self.sess.run(
                [self.grad_total_loss_op, self.loss_no_reg, self.total_loss, self.accuracy_op], 
                feed_dict=self.all_train_feed_dict)

            test_loss_val, test_acc_val = self.sess.run(
                [self.loss_no_reg, self.accuracy_op], 
                feed_dict=self.all_test_feed_dict)



        print('(All values are taken average among all data)')
        print('Train loss (with regression): %.2f' % loss_val)
        print('Train loss (without regression): %.2f' % loss_no_reg_val)
        print('Test loss (without regression): %.2f' % test_loss_val)

        print()

        print('Train accuracy:  %.2f' % train_acc_val)
        print('Test accuracy: %.2f' % test_acc_val)

        print()

        print('Norm of gradients (loss to params): %.2f' % np.linalg.norm(np.concatenate(grad_loss_val)))
        print('Norm of the parameters: %.2f' % np.linalg.norm(np.concatenate(params_val)))

        print()

    def retrain(self, num_steps, feed_dict):        
        # run train_op for num_steps number of times with feed_dict
        for step in range(num_steps):
            self.sess.run(self.train_op, feed_dict=feed_dict)


    def update_learning_rate(self, step):
        # with Args:step calculate which epoch this is, and using decay_epoch (a length 2 list)
        # determine if the multiplier applied to initial_learning_rate is 1, 0.1 or 0.01
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
              iter_to_switch_to_batch=20000, 
              iter_to_switch_to_sgd=40000,
              save_checkpoints=True, verbose=True):
        # 3 training mode, specified by iter_to_switch_to_batch and iter_to_switch_to_sgd:
        #   mini-batch training:    fill_feed_dict_with_batch, sess run train_op & total_loss
        #   batch training:         all_train_feed_dict,       sess run train_op & total_loss
        #   sgd training:           all_train_feed_dict,       sess run train_sgd_op & total_loss
        if verbose: print('Training for %s steps' % num_steps)

        sess = self.sess            

        for step in range(num_steps):
            self.update_learning_rate(step)

            start_time = time.time()

            if step < iter_to_switch_to_batch:                
                feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train)
                _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)
                
            elif step < iter_to_switch_to_sgd:
                feed_dict = self.all_train_feed_dict          
                _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)

            else: 
                feed_dict = self.all_train_feed_dict          
                _, loss_val = sess.run([self.train_sgd_op, self.total_loss], feed_dict=feed_dict)          

            duration = time.time() - start_time

            if verbose:
                if step % 1000 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.8f (%.3f sec)' % (step, loss_val, duration))

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 100000 == 0 or (step + 1) == num_steps:
                if save_checkpoints: self.saver.save(sess, self.checkpoint_file, global_step=step)
                if verbose: self.print_model_eval()


    def load_checkpoint(self, iter_to_load, do_checks=True):
        # restore the checkpoints (model) saved at iteration specified by iter_to_load
        # and print the model evaluation at that time by calling print_model_eval()
        checkpoint_to_load = "%s-%s" % (self.checkpoint_file, iter_to_load) 
        self.saver.restore(self.sess, checkpoint_to_load)

        if do_checks:
            print('Model %s loaded. Sanity checks ---' % checkpoint_to_load)
            self.print_model_eval()


    def get_train_op(self, total_loss, global_step, learning_rate):
        # run adam optimizer, return optimizer.minimize
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op


    def get_train_sgd_op(self, total_loss, global_step, learning_rate=0.001):
        # run gradient descent optimizer, return optimizer.minimize
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op


    def get_accuracy_op(self, logits, labels):
        """
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


    def loss(self, logits, labels):
        """
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
            indiv_loss_no_reg:  cross entropy loss of single instances;
            loss_no_reg:        average cross entropy loss across the whole batch;
            total_loss:         sum of average cross entropy batch loss of all
                                training points so far.
        """
        labels = tf.one_hot(labels, depth=self.num_classes)
        cross_entropy = - tf.reduce_sum(tf.multiply(labels, tf.nn.log_softmax(logits)),
                                        reduction_indices=1)
        # losses of single instances
        indiv_loss_no_reg = cross_entropy

        # loss of the entire batch
        loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.add_to_collection('losses', loss_no_reg)

        # loss of all training so far
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return total_loss, loss_no_reg, indiv_loss_no_reg


    def adversarial_loss(self, logits, labels):
        """
        Calculates:
            log(sum(exp(value_of_logits_of_all_the_negative_labels_judged_by_NN))) -
            log(sum(exp(value_of_all_logits)))
        Args:
            logits: tensor, float, [batch_size, num_classes]
            labels: tensor, int, [batch_size], with values in range [0, num_classes)
        Return:
            indiv_adversarial_loss: cross entropy loss of single instances;
            adversarial_loss: average cross entropy across the whole batch
        """
        labels = tf.one_hot(labels, depth=self.num_classes)        
        wrong_labels = (labels - 1) * -1 # Flips 0s and 1s
        wrong_labels_bool = tf.reshape(tf.cast(wrong_labels, tf.bool), [-1, self.num_classes])
        wrong_logits = tf.reshape(tf.boolean_mask(logits, wrong_labels_bool),
                                  [-1, self.num_classes - 1])
        # boolean_mask:
        #
        #   tensor = [0, 1, 2, 3]
        #   mask = np.array([True, False, True, False])
        #   boolean_mask(tensor, mask)  # [0, 2]

        indiv_adversarial_loss = tf.reduce_logsumexp(wrong_logits, reduction_indices=1) - \
                                 tf.reduce_logsumexp(logits, reduction_indices=1)
        # reduce_logsumexp: log(sum(exp(x)))
        #
        #   a = tf.constant([[0,0,0],
        #                    [0,0,0]])
        #   a.shape() = (2,3)
        #   tf.reduce_logsumexp(a, reduction_indices=1) --> [log(3), log(3)]
        #   tf.reduce_logsumexp(a)                      --> log(6)
        adversarial_loss = tf.reduce_mean(indiv_adversarial_loss)
        
        return adversarial_loss, indiv_adversarial_loss #, indiv_wrong_prob


    def update_feed_dict_with_v_placeholder(self, feed_dict, vec):
        # recall v_placeholder is a placeholder of type tf.float32 and shape of each
        # set of parameters.
        #
        # for now we assume that feed_dict has a slot for each of these sets of parameters
        # and vec are their "new" values we want to assign to the feed_dict.
        for pl_block, vec_block in zip(self.v_placeholder, vec):
            feed_dict[pl_block] = vec_block        
        return feed_dict



################################################ HVP ################################################
    """
    the main method in HVP part is get_inverset_hvp, the rest are helper functions that
    can be ignored when using the class genericNeuralNet
    """
    def get_inverse_hvp(self, v, approx_type='cg', approx_params=None, verbose=True):
        assert approx_type in ['cg', 'lissa']
        if approx_type == 'lissa':
            return self.get_inverse_hvp_lissa(v, **approx_params)
        elif approx_type == 'cg':
            return self.get_inverse_hvp_cg(v, verbose)


    def get_inverse_hvp_lissa(self, v, 
                              batch_size=None,
                              scale=10, damping=0.0, num_samples=1, recursion_depth=10000):
        inverse_hvp = None
        print_iter = recursion_depth / 10

        # for each sample(t) we want to investigate, do recursion update for r times.
        # note that rt = O(n), where n is the total number of training points
        for i in range(num_samples):
            current_estimate = v
            for j in range(recursion_depth):
                # prepare for hessian vector evaluation by populating feed_dict with
                # training data of this batch and the cur_estimates (parameter placeholders)
                feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, batch_size=batch_size)
                feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, current_estimate)

                # running hessian_vector calls the hessian_vector_product in hessians.py, which
                # multiplies v by the derivative of total loss wrt each parameter
                hessian_vector_val = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
                # Update: current_estimate = v + (I - Hessian_at_x) * current_estimate
                current_estimate = [a + (1-damping) * b - c/scale for (a,b,c) in
                                zip(v, current_estimate, hessian_vector_val)]

                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    print("Recursion at depth %s: norm is %.4lf" %
                          (j, np.linalg.norm(np.concatenate(current_estimate))))
                    feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, current_estimate)

            if inverse_hvp is None:
                inverse_hvp = [b/scale for b in current_estimate]
            else:
                inverse_hvp = [a + b/scale for (a, b) in zip(inverse_hvp, current_estimate)]

        inverse_hvp = [a/num_samples for a in inverse_hvp]


        return inverse_hvp
  
    
    def minibatch_hessian_vector_val(self, v):

        num_examples = self.num_train_examples
        if self.mini_batch == True:
            batch_size = 100
            assert num_examples % batch_size == 0
        else:
            batch_size = self.num_train_examples

        num_iter = int(num_examples / batch_size)

        self.reset_datasets()
        hessian_vector_val = None
        for i in range(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, batch_size=batch_size)
            feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, v)
            hessian_vector_val_temp = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
            if hessian_vector_val is None:
                hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
            else:
                hessian_vector_val = [a + (b / float(num_iter)) for (a,b) in
                                      zip(hessian_vector_val, hessian_vector_val_temp)]
        hessian_vector_val = [a + self.damping * b for (a,b) in zip(hessian_vector_val, v)]
        return hessian_vector_val


    def get_fmin_loss_fn(self, v):

        def get_fmin_loss(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)
        return get_fmin_loss


    def get_fmin_grad_fn(self, v):
        def get_fmin_grad(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))
            
            return np.concatenate(hessian_vector_val) - np.concatenate(v)
        return get_fmin_grad


    def get_fmin_hvp(self, x, p):
        hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(p))

        return np.concatenate(hessian_vector_val)


    def get_cg_callback(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        
        def fmin_loss_split(x):
            # difference to fmin_loss_fn(): it's splitted
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x), -np.dot(np.concatenate(v), x)

        def cg_callback(x):
            # x is current params
            v = self.vec_to_list(x)
            idx_to_remove = 5

            single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)      
            train_grad_loss_val = self.sess.run(self.grad_total_loss_op,  # d(total loss)/d(params)
                                                feed_dict=single_train_feed_dict)
            predicted_loss_diff = np.dot(np.concatenate(v),
                                         np.concatenate(train_grad_loss_val)) / self.num_train_examples

            if verbose:
                print('Function value: %.2f' % fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print('Split function value: %.2f, %.2f' % (quad, lin))
                print('Predicted influence of train_idx %s: %.5f' % (idx_to_remove, predicted_loss_diff))
                print()
        return cg_callback


    def get_inverse_hvp_cg(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        fmin_grad_fn = self.get_fmin_grad_fn(v)
        cg_callback = self.get_cg_callback(v, verbose)

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=np.concatenate(v),
            fprime=fmin_grad_fn,
            fhess_p=self.get_fmin_hvp,
            callback=cg_callback,
            avextol=1e-8,
            maxiter=100) 

        return self.vec_to_list(fmin_results)
######################################################################################################

    def get_test_grad_loss_no_reg_val(self, test_indices, batch_size=100, loss_type='normal_loss'):
        # A helper function for get_influence_on_test_loss:
        #
        # The function takes a list of test indices and calculates d(loss on test set)/d(parameters)
        # (can be normal loss or adversarial loss), take average. Returns single value. The output
        # of this function is needed for HVP calculation.
        if loss_type == 'normal_loss':
            op = self.grad_loss_no_reg_op
        elif loss_type == 'adversarial_loss':
            op = self.grad_adversarial_loss_op
        else:
            raise ValueError('Loss must be specified')

        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / batch_size))

            test_grad_loss_no_reg_val = None
            for i in range(num_iter):
                start = i * batch_size
                end = int(min((i+1) * batch_size, len(test_indices)))

                test_feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.test,
                                                                  test_indices[start:end])

                temp = self.sess.run(op, feed_dict=test_feed_dict)

                if test_grad_loss_no_reg_val is None:
                    test_grad_loss_no_reg_val = [a * (end-start) for a in temp]
                else:
                    test_grad_loss_no_reg_val = [a + b * (end-start) for (a, b) in
                                                 zip(test_grad_loss_no_reg_val, temp)]

            test_grad_loss_no_reg_val = [a/len(test_indices) for a in test_grad_loss_no_reg_val]

        else:
            test_grad_loss_no_reg_val = self.minibatch_mean_eval([op], self.data_sets.test)[0]
        return test_grad_loss_no_reg_val


    def get_influence_on_test_loss(self,
                                   test_indices,
                                   train_indices,
                                   approx_type='cg',
                                   approx_params=None,
                                   force_refresh=True,
                                   test_description=None,
                                   loss_type='normal_loss',
                                   X=None,
                                   Y=None):


        # First we use the get_test_grad_loss_no_reg_val function to calculate
        # d(loss on test set)/d(parameters), with test set specified by list
        # test_indices. Using the return value of the function, we then calculate
        # the HVP wrt all entries in test_indices and save into an .npz file labeled
        # by these indices (or load if exist).
        #
        # We then train the model with each one of the single training datapoint
        # specified by train_indices (or X, Y), take the gradient of the training loss
        # w.r.t. parameters, and multiply each one by inverse HVP of test_indices, which
        # gives us a list, with its entries corresponding to the influence of each training
        # datapoints in train_indices

        # Args:
        #   test_indices:       A list of test indices we want to test the influence on
        #   train_indices:      A list of train indices we want to test the influence for
        #   approx_type:        'cg' or 'lissa'
        #   approx_params:      If 'lissa' provide batch_size, (optional arguments scale,
        #                       damping, num_samples, recursion_depth), otherwise None
        #   force_refresh:      Whether to recalculate the inverse HVP of existed .npz
        #   test_description:   When test_indices is too long, you might not want to use
        #                       this list as your .npz file name. in this case, put in
        #                       a description instead for simplicity
        #   loss_type:          'normal_loss' or 'adversarial_loss'
        #   X, Y:               When you want to specify the actual training datapoints you
        #                       wish to test the influence for (instead of their indices
        #                       in dataset), populate X, Y with training points.
        #                       X: [num_examples, datalength]
        #                       Y: [num_examples] (labels)

        # Returns:
        #   influences:   A list of influence evaluated for each training data point
        #                 against all testing indices.
        #                           len(influences) = len(train_indices)
        if train_indices is None:
            if (X is None) or (Y is None):
                raise ValueError('X and Y must be specified if using phantom points.')
            if X.shape[0] != len(Y):
                raise ValueError('X and Y must have the same length.')
        else:
            if (X is not None) or (Y is not None):
                raise ValueError('X and Y cannot be specified if train_idx is specified.')

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices,
                                                                       loss_type=loss_type)

        print('Norm of test gradients (loss to params): %s' %
              np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz'
                                       % (self.model_name, approx_type, loss_type, test_description))
                                        # dogfish_inception    cg      normal_loss      [462]
        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            print('Loaded inverse HVP from %s' % approx_filename)
        else:
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            print('Saved inverse HVP to %s' % approx_filename)
            duration = time.time() - start_time
            print('Inverse HVP took %.2f sec' % duration)



        start_time = time.time()
        if train_indices is None:

            num_to_remove = len(Y)
            influences = np.zeros([num_to_remove])
            for counter in np.arange(num_to_remove):
                single_train_feed_dict = self.fill_feed_dict_manual(X[counter, :],
                                                                    [Y[counter]])
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op,
                                                    feed_dict=single_train_feed_dict)
                influences[counter] = np.dot(np.concatenate(inverse_hvp),
                                                       np.concatenate(train_grad_loss_val)) \
                                                / self.num_train_examples

        else:            
            num_to_remove = len(train_indices)
            influences = np.zeros([num_to_remove])
            for counter, idx_to_remove in enumerate(train_indices):
                single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train,
                                                                         idx_to_remove)
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op,
                                                    feed_dict=single_train_feed_dict)
                influences[counter] = np.dot(np.concatenate(inverse_hvp),
                                                       np.concatenate(train_grad_loss_val))\
                                                / self.num_train_examples
        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

        return influences



    def find_eigvals_of_hessian(self, num_iter=100, num_prints=10):
        # Find the largest and smallest eigenvalues of hessian vector product matrix
        # by doing a power iteration (the hessian vector product seems to be random?)

        # Setup        
        print_iterations = num_iter / num_prints

        # Initialize starting vector: create a vector made up by random values with the
        # exact same shape of d(total loss)/d(parameters)
        feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, 0)
        grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=feed_dict)
        initial_v = []
        for a in grad_loss_val:
            initial_v.append(np.random.random(a.shape))        
        initial_v, norm_val = normalize_vector(initial_v) # divided by normal value of vector

        # Do power iteration to find largest and smallest eigenvalue
        print('Starting power iteration to find largest and smallest eigenvalue...')
        print('Initial average value of of the eigenvlaue is %s' % norm_val)

        # Do power iteration to find smallest eigenvalue
        largest_eig = norm_val
        current_estimate = initial_v
        for i in range(num_iter):          
            current_estimate, norm_val = normalize_vector(current_estimate)
            hessian_vector_val = self.minibatch_hessian_vector_val(current_estimate)
                                # call hessian_vector_product defined in influence\hessians.py
            new_current_estimate = [a - largest_eig * b for
                                    (a, b) in
                                    zip(hessian_vector_val, current_estimate)]

            if i % print_iterations == 0:
                print(-norm_val + largest_eig)
                dotp = np.dot(np.concatenate(new_current_estimate), np.concatenate(current_estimate))
                print("dot: %s" % dotp)
            current_estimate = new_current_estimate

        smallest_eig = -norm_val + largest_eig
        assert dotp < 0, "Eigenvalue calc failed to find largest eigenvalue"

        print('Largest eigenvalue is %s' % largest_eig)
        print('Smallest eigenvalue is %s' % smallest_eig)
        return largest_eig, smallest_eig


    def get_grad_of_influence_wrt_input(self,
                                        train_indices,
                                        test_indices,
                                        approx_type='cg',
                                        approx_params=None,
                                        force_refresh=True,
                                        verbose=True,
                                        test_description=None,
                                        loss_type='normal_loss'):
        """
        If the loss goes up when you remove a point, then it was a helpful point.
        So positive influence = helpful.
        If we move in the direction of the gradient, we make the influence even more positive, 
        so even more helpful.
        Thus if we want to make the test point more wrong, we have to move in the opposite direction.
        """

        # Calculate v_placeholder (gradient of loss at test point)
        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices,
                                                                       loss_type=loss_type)

        if verbose:
            print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))
        
        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' %
                                       (self.model_name, approx_type, loss_type, test_description))
        
        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            if verbose: print('Loaded inverse HVP from %s' % approx_filename)
        else:            
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params,
                verbose=verbose)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            if verbose: print('Saved inverse HVP to %s' % approx_filename)            
        
            duration = time.time() - start_time
            if verbose: print('Inverse HVP took %s sec' % duration)

        grad_influence_wrt_input_val = None

        for counter, train_idx in enumerate(train_indices):
            # Put in the train example in the feed dict
            grad_influence_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, train_idx)
            self.update_feed_dict_with_v_placeholder(grad_influence_feed_dict, inverse_hvp)
            # fill the self.v_placeholder (not feed dict) with inverse HVP

            # Run the gradient op with the feed dict, calculate the gradient of influence with
            # respect to input (specified by train_idx)
            current_grad_influence_wrt_input_val = self.sess.run(self.grad_influence_wrt_input_op,
                                                                 feed_dict=grad_influence_feed_dict)[0][0, :]
            
            if grad_influence_wrt_input_val is None:
                grad_influence_wrt_input_val = np.zeros([len(train_indices),
                                                         len(current_grad_influence_wrt_input_val)])

            grad_influence_wrt_input_val[counter, :] = current_grad_influence_wrt_input_val
        # returns a list of gradient influence with respect to inputs
        return grad_influence_wrt_input_val


    def update_train_x(self, new_train_x):
        assert np.all(new_train_x.shape == self.data_sets.train.x.shape)
        new_train = DataSet(new_train_x, np.copy(self.data_sets.train.labels))
        self.data_sets = base.Datasets(train=new_train, validation=self.data_sets.validation, test=self.data_sets.test)
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)                
        self.reset_datasets()


    def update_train_x_y(self, new_train_x, new_train_y):
        new_train = DataSet(new_train_x, new_train_y)
        self.data_sets = base.Datasets(train=new_train, validation=self.data_sets.validation, test=self.data_sets.test)
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)                
        self.num_train_examples = len(new_train_y)
        self.reset_datasets()        


    def update_test_x_y(self, new_test_x, new_test_y):
        new_test = DataSet(new_test_x, new_test_y)
        self.data_sets = base.Datasets(train=self.data_sets.train, validation=self.data_sets.validation, test=new_test)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test)                
        self.num_test_examples = len(new_test_y)
        self.reset_datasets()        

