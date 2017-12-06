import matplotlib.image as img
import glob
import math
import numpy as np
import random

def load_data(data_dir, split=1.0):
    """ Returns X,Y partioned into test and train sets as row vectors."""
    n = 123 * 41
    positive_examples = glob.glob(data_dir + '/positive_set/*.png')
    negative_examples = glob.glob(data_dir + '/negative_set/*.png')
    x = np.zeros((n, len(positive_examples) + len(negative_examples)))
    for k in range(len(positive_examples)):
        x[:,k] = np.reshape(img.imread(positive_examples[k]), (n,))
    for k in range(len(negative_examples)):
        x[:, k+len(positive_examples)] = np.reshape(img.imread(negative_examples[k]), (n,))
    y = np.concatenate((np.ones((1,len(positive_examples))), np.zeros((1,len(negative_examples)))), axis=1)
    arr = np.concatenate((x,y), axis=0)
    part_train = int(math.floor(split * arr.shape[1]))
    np.random.shuffle(arr.T)
    x_train = arr[0:-1,0:part_train]
    y_train = arr[-1,0:part_train]
    x_test = arr[0:-1,part_train:]
    y_test = arr[-1,part_train:]
    if split == 1.0:
        return (x_train.T, y_train.T)
    else:
        return (x_train.T, y_train.T),(x_test.T,y_test.T)


