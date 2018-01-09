from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import influence.experiments as experiments
from influence.all_CNN_c import All_CNN_C
from scripts.load_cifar10 import load_cifar10


data_sets = load_cifar10(num_train_ex_per_class=4608,
                         num_test_ex_per_class=256)

num_classes = 10
input_side = 32
input_channels = 3
input_dim = input_side * input_side * input_channels
weight_decay = 0.01
batch_size = 256

initial_learning_rate = 0.0001
decay_epochs = [10000, 20000]
hidden1_units = 48
hidden2_units = 96
hidden3_units = 192
conv_patch_size = 3
remove_type = 'neginf'
approx_type = 'cg'
num_steps = 100000

tf.reset_default_graph()

model = All_CNN_C(
    input_side=input_side,
    input_channels=input_channels,
    conv_patch_size=conv_patch_size,
    hidden1_units=hidden1_units,
    hidden2_units=hidden2_units,
    hidden3_units=hidden3_units,
    weight_decay=weight_decay,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output',
    log_dir='log',
    model_name='cifar10_all_cnn_c')


# model.train(num_steps=num_steps,
#             iter_to_switch_to_batch=500000,
#             iter_to_switch_to_sgd=1000000)


test_idx = 558
iter_to_load = num_steps - 1
actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
    model,
    test_idx=test_idx,
    iter_to_load=iter_to_load,
    num_to_remove=100,
    remove_type=remove_type,
    force_refresh=True,
    approx_type=approx_type,
    approx_params={'batch_size':1})


np.savez(
    'output/mnist_small_all_cnn_c_iter-500k_retraining-100.npz',
    actual_loss_diffs=actual_loss_diffs,
    predicted_loss_diffs=predicted_loss_diffs,
    indices_to_remove=indices_to_remove
    )
