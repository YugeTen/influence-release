import os
from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np
from subprocess import call
from keras.preprocessing import image
from influence.dataset import DataSet
from influence.inception_v3 import preprocess_input
from sklearn.utils import shuffle

BASE_DIR = 'data' # TODO: change

def fill(X, Y, idx, label, img_path, img_size):
    """Args:
        X:      image holder, initialised with np.zeros,
                [num_examples, img_size, img_size, num_channels]
        Y:      label holder, initialised with np.zeros, [num_examples]
        idx:    "counter", specifies which image it is in the whole batch
        label:  index of class
    """

    img = image.load_img(img_path, target_size=(img_size, img_size))
    x = image.img_to_array(img) #Converts image to 3D numpy array
    X[idx, ...] = x # X[1, ...] <==> X[1, :, :, :]
    Y[idx] = label

def extract_and_rename_animals():
    """This function is not used in project"""
    class_maps = [
        ('dog', 'n02084071'),
        ('cat', 'n02121808'),
        ('bird', 'n01503061'),
        ('fish', 'n02512053'),
        ('horse', 'n02374451'),
        ('monkey', 'n02484322'),
        ('zebra', 'n02391049'),
        ('panda', 'n02510455'),
        ('lemur', 'n02496913'),
        ('wombat', 'n01883070'),
        ]
    for class_string, class_id in class_maps:
        class_dir = os.path.join(BASE_DIR, class_string)
        print(class_dir) # typically data/dog
        call('mkdir %s' % class_dir, shell=True)
        call('tar -xf %s.tar -C %s' % (os.path.join(BASE_DIR, class_id), class_dir), shell=True)

        for filename in os.listdir(class_dir):
            file_idx = filename.split('_')[1].split('.')[0]
            src_filename = os.path.join(class_dir, filename)
            dst_filename = os.path.join(class_dir, '%s_%s.JPEG' % (class_string, file_idx))
            os.rename(src_filename, dst_filename)


def load_animals(num_train_ex_per_class=300,
                 num_test_ex_per_class=100,
                 num_valid_ex_per_class=0,
                 classes=None):
    """Either load animals from an .npz file (if already created)
    or create the .npz file from raw images"""

    num_channels = 3
    img_size = 299

    # The 2 if-else statements are just creating npz filename strings
    # that contains:
    #   name of class
    #   number of training instances
    #   number of testing instances
    #   number of validation instances
    if num_valid_ex_per_class == 0:
        valid_str = ''
    else:
        valid_str = '_valid-%s' % num_valid_ex_per_class

    if classes is None:
        classes = ['dog', 'cat', 'bird', 'fish', 'horse',
                   'monkey', 'zebra', 'panda', 'lemur', 'wombat']
        data_filename = os.path.join(BASE_DIR,
                                     'dataset_train-%s_test-%s%s.npz'
                                     % (num_train_ex_per_class,
                                        num_test_ex_per_class,
                                        valid_str))
    else:
        data_filename = os.path.join(BASE_DIR,
                                     'dataset_%s_train-%s_test-%s%s.npz'
                                     % ('-'.join(classes),
                                        num_train_ex_per_class,
                                        num_test_ex_per_class,
                                        valid_str))

    num_classes = len(classes)
    num_train_examples = num_train_ex_per_class * num_classes
    num_test_examples = num_test_ex_per_class * num_classes
    num_valid_examples = num_valid_ex_per_class * num_classes
    num_ex_per_class = num_train_ex_per_class+num_valid_ex_per_class+num_test_ex_per_class
    num_examples = num_train_examples + num_test_examples + num_valid_examples


    if os.path.exists(data_filename):
        print ('Loading data from pre-existed .npz file...')
        f = np.load(data_filename)
        X_train = f['X_train']
        X_test = f['X_test']
        Y_train = f['Y_train']
        Y_test = f['Y_test']
        X_valid = f['X_valid'] if 'X_valid' in f else None
        Y_valid = f['Y_valid'] if 'Y_valid' in f else None

    else:
        print('Creating .npz file from raw images...')
        # initialization
        X = np.zeros([num_examples, img_size, img_size, num_channels])
        Y = np.zeros([num_examples])

        for class_idx, class_string in enumerate(classes):
            print('class: %s' % class_string)
            i = 0
            num_filled = 0
            # no. of images that has been loaded in X_train
            while num_filled < num_ex_per_class:
                img_path = os.path.join(BASE_DIR,
                                        '%s/%s_%s.JPEG'%(class_string,
                                                         class_string,
                                                         i))
                if os.path.exists(img_path):
                    fill(X,
                         Y,
                         num_filled+(num_ex_per_class*class_idx),
                         class_idx,
                         img_path,
                         img_size)
                    num_filled += 1
                i += 1

        X, Y = shuffle(X,Y)

        X_train = X[0:num_train_examples,...]
        Y_train = Y[0:num_train_examples, ...]
        X_test = X[num_train_examples:num_train_examples+num_test_examples,...]
        Y_test = Y[num_train_examples:num_train_examples+num_test_examples,...]
        X_valid = X[num_train_examples+num_test_examples:-1,...]
        Y_valid = Y[num_train_examples+num_test_examples:-1,...]

        # preprocess input with Inception V3 config
        X_train = preprocess_input(X_train)
        X_test = preprocess_input(X_test)
        X_valid = preprocess_input(X_valid)

        np.savez_compressed(data_filename,
                            X_train=X_train,
                            Y_train=Y_train,
                            X_test=X_test,
                            Y_test=Y_test,
                            X_valid=X_valid,
                            Y_valid=Y_valid)

    train = DataSet(X_train, Y_train) # see def of DataSet in influence/dataset
    test = DataSet(X_test, Y_test)
    validation = DataSet(X_valid, Y_valid) if X_valid and Y_valid else None

    # base: base utilities of tensorflow for loading datasets
    return base.Datasets(train=train, validation=validation, test=test)


def load_koda():
    num_channels = 3
    img_side = 299

    data_filename = os.path.join(BASE_DIR, 'dataset_koda.npz')

    if os.path.exists(data_filename):
        print('Loading Koda from disk...')
        f = np.load(data_filename)
        X = f['X']
        Y = f['Y']
    else:
        # Returns all class 0
        print('Reading Koda from raw images...')

        image_files = [image_file for image_file in os.listdir(os.path.join(BASE_DIR, 'koda')) if
                       (image_file.endswith('.jpg'))]

        num_examples = len(image_files)
        X = np.zeros([num_examples, img_side, img_side, num_channels])
        Y = np.zeros([num_examples])

        class_idx = 0
        for counter, image_file in enumerate(image_files):
            img_path = os.path.join(BASE_DIR, 'koda', image_file)
            fill(X, Y, counter, class_idx, img_path, img_side)

        X = preprocess_input(X)

        np.savez(data_filename, X=X, Y=Y)

    return X, Y


def load_dogfish_with_koda():
    classes = ['dog', 'fish']
    X_test, Y_test = load_koda()

    data_sets = load_animals(num_train_ex_per_class=900,
                             num_test_ex_per_class=300,
                             num_valid_ex_per_class=0,
                             classes=classes)
    train = data_sets.train
    validation = data_sets.validation
    test = DataSet(X_test, Y_test)

    return base.Datasets(train=train, validation=validation, test=test)


def load_dogfish_with_orig_and_koda():
    classes = ['dog', 'fish']
    X_test, Y_test = load_koda()
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    data_sets = load_animals(num_train_ex_per_class=900,
                             num_test_ex_per_class=300,
                             num_valid_ex_per_class=0,
                             classes=classes)
    train = data_sets.train
    validation = data_sets.validation

    test = DataSet(
        np.concatenate((data_sets.test.x, X_test), axis=0),
        np.concatenate((data_sets.test.labels, Y_test), axis=0))

    return base.Datasets(train=train, validation=validation, test=test)





































