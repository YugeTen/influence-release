import os

from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np
import IPython

from subprocess import call

from keras.preprocessing import image
from sklearn.utils import shuffle

from influence.dataset import DataSet
from influence.inception_v3 import preprocess_input

from scripts.load_animals import fill
import csv

BASE_DIR = '/home/shi144/influence-release/scripts/data'



def load_cifar10(
        num_train_ex_per_class=300,
        num_test_ex_per_class=100,
        num_valid_ex_per_class=0,
        classes=None):
    num_channels = 3
    img_side = 32

    if num_valid_ex_per_class == 0:
        valid_str = ''
    else:
        valid_str = '_valid-%s' % num_valid_ex_per_class

    if classes is None:
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
        data_filename = os.path.join(BASE_DIR, 'cifar10_train-%s_test-%s%s.npz'%
                                     (num_train_ex_per_class, num_test_ex_per_class,valid_str))
    else:
        data_filename = os.path.join(BASE_DIR, 'cifar10_%s_train-%s_test-%s%s.npz' %
                                     ('-'.join(classes),
                                     num_train_ex_per_class,
                                     num_test_ex_per_class,
                                     valid_str))

    num_classes = len(classes)
    num_train_examples = num_train_ex_per_class*num_classes
    num_test_examples = num_test_ex_per_class*num_classes
    num_valid_examples = num_valid_ex_per_class*num_classes
    num_examples = num_train_examples + num_test_examples + num_valid_examples

    if os.path.exists(data_filename):
        print('Loading dataset from disk...')
        f = np.load(data_filename)
        x_train = f['x_train']
        x_test = f['x_test']
        y_train = f['y_train']
        y_test = f['y_test']

        if 'x_valid' in f:
            x_valid = f['x_valid']
        else:
            x_valid = None

        if 'y_valid' in f:
            y_valid = f['y_valid']
        else:
            y_valid = None

    else:
        print('Loading dataset from raw images...')
        x = np.zeros([num_examples, img_side, img_side, num_channels])
        y = np.zeros([num_examples])

        num_ex_per_class = num_train_ex_per_class + num_test_ex_per_class + num_valid_ex_per_class

        for class_idx, class_string in enumerate(classes):
            print('class: %s' % class_string)

            idx_strings=[]
            labels=[]
            with open('/home/shi144/Desktop/trainLabels.csv') as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    idx_strings.append(row[0])
                    labels.append(row[1])
            idx_strings = np.asarray(idx_strings)
            labels = np.asarray(labels)
            class_idx_strings = idx_strings[np.where(labels==class_string)[0]]

            num_filled = 0
            while num_filled < num_ex_per_class:
                img_path = os.path.join(BASE_DIR,
                                            'cifar10',
                                            class_idx_strings[num_filled]+'.png')
                if os.path.exists(img_path):

                    fill(X=x,
                         Y=y,
                         idx=num_filled+(num_ex_per_class*class_idx),
                         label=class_idx,
                         img_path=img_path,
                         img_side=img_side)
                    num_filled += 1

        x, y = shuffle(x,y)
        x_train = x[0:num_train_examples, ...]
        y_train = y[0:num_train_examples, ...]
        x_test = x[num_train_examples:num_train_examples+num_test_examples, ...]
        y_test = y[num_train_examples:num_train_examples+num_test_examples, ...]
        x_valid = x[num_train_examples+num_test_examples:-1, ...]
        y_valid = y[num_train_examples+num_test_examples:-1, ...]
        np.savez_compressed(data_filename,
                            x_train=x_train,
                            y_train=y_train,
                            x_test=x_test,
                            y_test=y_test,
                            x_valid=x_valid,
                            y_valid=y_valid)

    train = DataSet(x_train, y_train) # see def of DataSet in influence/dataset
    test = DataSet(x_test, y_test)
    validation = DataSet(x_valid, y_valid) if x_valid and y_valid else None

    # base: base utilities of tensorflow for loading datasets
    return base.Datasets(train=train, validation=validation, test=test)






