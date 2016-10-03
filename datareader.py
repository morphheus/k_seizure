"""Module to read and write the .mat data in batches"""
# translation of the Matlab feature extractor 
# Credit: https://www.kaggle.com/deepcnn
import pdb
import sys
import os
import numpy as np
import pandas as pd
from math import *
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

#import pyeeg 
# pyeeg is the one that has very good fractal dimensions 
# computation but not installed here

# Globals
TRAIN_DIR_PATH = 'training_data/'
TRAIN_PATIENT_PATH_PREFIX = 'train_'
PATIENTS = [1]

def mat_to_data(path):
    """From matlab format to usable format"""
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata

def get_train_patient_list():
    """Returns the number of patients and their folder prefixes"""
    directories = os.listdir(TRAIN_DIR_PATH)
    patients = [int(s[s.index('_')+1:]) for s in directories]
    patients_paths = [TRAIN_DIR_PATH + x + '/' for x in directories]
    return patients, patients_paths

def get_train_data_paths(patient):
    """Returns a list of files for that patient"""
    paths = os.listdir(TRAIN_DIR_PATH + TRAIN_PATIENT_PATH_PREFIX + str(patient))
    tmp = [(int(s.split('_')[1]), s) for s in paths]
    tmp.sort()
    sorted_paths = list(zip(*tmp))[1]
    targets = [int(s[-1*s[::-1].index('_'):-4]) for s in sorted_paths]
    # Add the path prefix
    out = [TRAIN_DIR_PATH + TRAIN_PATIENT_PATH_PREFIX + str(patient) + '/' + s for s in sorted_paths]
    return out, targets

def stratified_folds(X, Y, folds=7, shuffle_split=False):
    """Returns the indicies for training folds and test sets. Note that the shuffle option corresponds to a ShuffleSplit cross-validator""" 
    sss = StratifiedShuffleSplit(n_splits=folds, test_size=0.2) if shuffle_split else StratifiedKFold(n_splits=folds, shuffle=False)
    for train_index, test_index in sss.split(X,Y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]




if __name__ == '__main__':
    pass
    #paths, targets = get_train_data_paths(1)
    #[print(str(t) + '  ' + s) for s,t in zip(paths, targets)]
    X = np.arange(20)
    Y = np.append(np.ones(5), np.zeros(15))
    stratified_folds(X,Y,folds=4, shuffle_split=False)



