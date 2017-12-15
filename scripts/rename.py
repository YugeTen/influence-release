from __future__ import print_function, division, absolute_import

import hashlib
import os.path
import random
import re
import pickle
import gzip
import inspect
import argparse
import sys
from datetime import datetime
import struct
import tarfile




import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy.matlib

from tensorflow.python.platform import gfile
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.util import compat
from sklearn.cluster import MiniBatchKMeans

################################## rename from .jpg to _JPEG ################################
# folder_dir = "/home/shi144/influence-release/scripts/data/"
# sub_dirs = [x[0] for x in gfile.Walk(folder_dir)]  # sub_dirs = ~/humanAction/walking,...
# # skip the root directory
# is_root_dir = True
# for sub_dir in sub_dirs:
#     if is_root_dir:
#         is_root_dir = False
#         continue
#     file_list = []
#     dir_name = os.path.basename(sub_dir)
#     if dir_name == folder_dir:
#         continue
#     file_glob = os.path.join(folder_dir, dir_name, '*.' + 'jpg')
#     file_list.extend(gfile.Glob(file_glob))
#
#
#     for file in file_list:
#         pre,mid,sub = file.rsplit('.')
#         newfile = pre+'_'+mid+'.JPEG'
#         print(file)
#         print(newfile)
#         os.rename(file,newfile)

############################### rename from cat to fish #################################
folder_dir = "/home/shi144/influence-release/scripts/data/cat/"
result = {}
sub_dirs = [x[0] for x in gfile.Walk(folder_dir)]  # sub_dirs = ~/humanAction/walking,...
sub_dir = sub_dirs[0]

print(sub_dir)
#
file_list = []
dir_name = os.path.basename(sub_dir)
file_glob = os.path.join(folder_dir, dir_name, '*.' + 'JPEG')
file_list.extend(gfile.Glob(file_glob))


for file in file_list:

    file_name = os.path.basename(file)
    print(file_name)
    new_file_name = file.replace('dog', 'cat')
    print(new_file_name)
    os.rename(file,new_file_name)

    #     newfile = file.replace('dog', 'cat')
    #     print(file)
    #     print(newfile)
    #     # os.rename(file,newfile)