from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import pandas as pd

import IPython
import tensorflow as tf

from influence.logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.experiments as experiments

from scripts.load_animals import load_animals
from scripts.load_mnist import load_mnist


data_sets = load_animals(num_train_ex_per_class=300,
                         num_test_ex_per_class=100,
                         classes=['dog', 'fish'])
# size: 299x299x3




num_classes = 2

input_dim = data_sets.train.x.shape[1]
weight_decay = 0.01
batch_size = 1400
initial_learning_rate = 0.001
keep_probs = None
max_lbfgs_iter = 5000
decay_epochs = [1000, 10000]
num_to_remove = 50
remove_type = 'neginf'

tf.reset_default_graph()


tf_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='mnist_logreg_lbfgs',
    num_to_remove=num_to_remove)

tf_model.train()

test_idx = 8
actual_loss_diffs, predicted_loss_diffs_cg, indices_to_remove = experiments.test_retraining(
    tf_model,
    test_idx=test_idx,
    iter_to_load=0,
    force_refresh=False,
    num_to_remove=num_to_remove,
    remove_type=remove_type,
    random_seed=0,
    approx_type='lissa',
    approx_params={'batch_size':1}
)
# #
# experiments.viz_top_influential_examples(tf_model, [123], 20)
# LiSSA
np.random.seed(17)
predicted_loss_diffs_lissa = tf_model.get_influence_on_test_loss(
    [test_idx],
    indices_to_remove,
    approx_type='lissa',
    approx_params={'scale':25,
                   'recursion_depth':5000,
                   'damping':0,
                   'batch_size':1,
                   'num_samples':10},
    force_refresh=True
)

np.savez(
    'output/mnist_logreg_lbfgs_retraining-500.npz',
    actual_loss_diffs=actual_loss_diffs,
    predicted_loss_diffs_cg=predicted_loss_diffs_cg,
    predicted_loss_diffs_lissa=predicted_loss_diffs_lissa,
    indices_to_remove=indices_to_remove
    )