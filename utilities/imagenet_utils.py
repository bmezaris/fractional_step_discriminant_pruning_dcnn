"""
Implementation of imagenet32 utility functions for the paper.

Based on code obtained from:
https://github.com/PatrykChrabaszcz/Imagenet32_Scripts
History
-------
DATE       | DESCRIPTION    | NAME              | ORGANIZATION |
16/01/2020 | first creation | Nikolaos Gkalelis | CERTH-ITI    |
"""


import os
import pickle
import _pickle as cPickle
import torch
import gzip
import numpy as np

def unpickle(file):
    dict = np.load(file + '.npz')
    return dict

def load_validation_data(data_folder, mean_image, img_size=32):
    test_file = os.path.join(data_folder, 'val_data')

    d = unpickle(test_file)
    x = d['data']
    y = d['labels']
    x = x / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = np.array([i-1 for i in y])

    # Remove mean (computed from training data) from images
    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    return dict(
        X_test=x,
        Y_test=y.astype('int32') )

def load_databatch(data_folder, idx, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    del d

    x = x/np.float32(255)
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    return dict(
        X_train= x[0:data_size, :, :, :],
        Y_train= np.array(y[0:data_size]).astype('int32'),
        mean=mean_image)