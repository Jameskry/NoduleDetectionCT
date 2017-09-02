#!/usr/bin/env python
"""
Builds a HDF5 data set for test, train and validation data
Run script as python build_hdf5_datasets.py $mode
where mode can be 'test', 'train', 'val'
"""

import sys
import numpy as np
import pandas as pd
import tflearn
from tflearn.data_utils import build_hdf5_image_dataset
import pickle
import h5py

# Check inputs
if len(sys.argv) < 4:
    raise ValueError('3 argument needed. train/test/val, positive_sample, negative_sample')
else:
    mode = sys.argv[1]
    n_pos_sample = int(sys.argv[2])
    n_neg_sample = int(sys.argv[3])
    if mode not in ['train', 'test', 'val'] :
        raise ValueError('Argument not recognized. Has to be train, test or val')

# Read data
X = pd.read_pickle('./data/' + mode + 'data.pickle')
y = pd.read_pickle('./data/' + mode + 'labels.pickle')

print X.shape

#take subset since we have small memory in local PC
positives = X[y==1].index
negatives = X[y==0].index

np.random.seed(40)

pos_sampled = np.random.choice(positives, n_pos_sample, replace=False)
neg_sampled = np.random.choice(negatives, n_neg_sample, replace=False)

X_sampled = X.loc[list(pos_sampled) + list(neg_sampled)]
y_sampled = y.loc[list(pos_sampled) + list(neg_sampled)]


dataset_file = './data/' + mode + 'datalabels.txt'

filenames = X_sampled.index.to_series().apply(lambda x: './data/' + mode + '/image_' + str(x) + '.jpg')

filenames = filenames.values.astype(str)
labels = y_sampled.values.astype(int)
data = np.zeros(filenames.size, dtype=[('col1', 'S36'), ('col2', int)])

data['col1'] = filenames
data['col2'] = labels

np.savetxt(dataset_file, data, fmt="%10s %d")  # datapath to label map

output = './data/' + mode + 'dataset_'+str(n_pos_sample)+'_'+str(n_neg_sample)+'.h5'

build_hdf5_image_dataset(dataset_file, image_shape=(512, 512),
                         mode='file', output_path=output, categorical_labels=True, normalize=True,
                         grayscale=True)









