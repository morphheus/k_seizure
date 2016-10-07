"""Module to read and write the .mat data in batches"""
import pdb
import sys
import os
import numpy as np
import pandas as pd
import resource
import warnings
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

#import pyeeg 
# pyeeg is the one that has very good fractal dimensions 
# computation but not installed here

# Globals
ROOT_DATA_DIR = 'data/'
TRAIN_DIR_PATH = 'training_data/'
TRAIN_PATIENT_PATH_PREFIX = 'train_'

TEST_DIR_PATH = 'testing_data/'
TEST_PATIENT_PATH_PREFIX = 'test_'
PATIENTS = [1]

def mat_to_data(path):
    """From matlab format to usable format"""
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata

def get_dirpath(train, preproc):
    """Builds the directorypath, which depends if the user wants to access the train/test, raw/preproc data"""
    dirpath = ROOT_DATA_DIR
    dirpath += TRAIN_DIR_PATH[:-1] if train else TEST_DIR_PATH[:-1]
    dirpath += '_preproc' if preproc else ''
    dirpath += '/'
    return dirpath

def get_preproc_path(rawpath, train):
    """From the raw filepath, returns the filepath corresponding to the preproc folder"""
    dirpath = get_dirpath(train, True)
    return dirpath + '/'.join(rawpath.split('/')[-2:])


def get_patient_list(train=True, preproc=True):
    """Returns the number of patients and their folder prefixes"""
    dirpath = get_dirpath(train, preproc)

    directories = os.listdir(dirpath)
    patients = [int(s[s.index('_')+1:]) for s in directories]
    patients_paths = [dirpath + x + '/' for x in directories]
    return patients, patients_paths

def get_data_paths(patient, train=True, preproc=True):
    """Returns a list of files for that patient"""
    dirpath = get_dirpath(train, preproc)

    paths = os.listdir(dirpath + TRAIN_PATIENT_PATH_PREFIX + str(patient))
    tmp = [(int(s.split('_')[1]), s) for s in paths]
    tmp.sort()
    sorted_paths = list(zip(*tmp))[1]
    targets = [int(s[-1*s[::-1].index('_'):-4]) for s in sorted_paths]
    # Add the path prefix
    out = [dirpath + TRAIN_PATIENT_PATH_PREFIX + str(patient) + '/' + s for s in sorted_paths]
    return out, targets

def stratified_folds(X, Y, folds=7, shuffle_split=False):
    """Returns the indicies for training folds and test sets. Note that the shuffle option corresponds to a ShuffleSplit cross-validator""" 
    sss = StratifiedShuffleSplit(n_splits=folds, test_size=0.2) if shuffle_split else StratifiedKFold(n_splits=folds, shuffle=False)
    for train_index, test_index in sss.split(X,Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        yield x_train, y_train ,x_test, y_test

def iterate_datafiles_maxmem(rawpathlist, loadfct, *args, maxmem=20, **kwargs):
    """Iterates through the datafiles keeping up to a maximum of data loaded in memory at a time
    This function should ideally be used with larger files. It will be slow with a lot of small files.
    pathlist: list of direct paths to the concerned files
    loadfct:  function called to load the concerned files. will call loadfct(path, *args, **kwargs)
    maxmem:   maximum memory to load, in MiB
    """
    maxmem_bytes = maxmem*1024**2
    # Obtain the size of all the files, as a ratio of the maximum allowed

    nested = True
    if type(rawpathlist[0]) == type(list()):
        pathlist = rawpathlist
        tmp = len(pathlist[0])
        for nest in pathlist:
            if len(nest) != tmp:
                raise Exception('The nested lists in pathlist have different lengths')
    else: # If not nested, nest it!
        nested = False
        pathlist = [rawpathlist]

    # Calculate the sizes
    tot_files = len(pathlist[0])
    sizes = np.zeros(tot_files).astype(np.float64)
    for k, sublist in enumerate(pathlist):
        sizes += np.array([os.stat(p).st_size for p in sublist]).astype(np.float64)
    sizes /= maxmem_bytes

    if True in (sizes>=1):
        warnings.warn('At least one file or set of files is larger than the specified maxmem')

    # Iteratively load
    k = 0
    while k < tot_files:
        prev_k = k
        batch_size = sizes[k]
        # Determine how sets of files we can take such that we won't bust the limit
        # A single too large file will be loaded alone
        while batch_size < 1:
            k += 1
            if k < tot_files:
                batch_size += sizes[k]
            else:
                break
        # Load data, yield it. Next time datalist is loaded, it will crush existing data, unless 
        # it was saved by the parent function call
        datalist = [[loadfct(path, *args, **kwargs) for path in sublist[prev_k:k]] for sublist in pathlist]
        if nested:
            for data in datalist:
                yield data
        else:
            for data in datalist[0]:
                yield data







def main():
    """Testing function"""
    pathlist, targets = get_train_data_paths(1, train=True, preproc=False)

    count = 10
    nested_pathlist = []
    for k in range(3):
        nested_pathlist.append(pathlist[k*count:(k+1)*count])


    pathlist = pathlist[:20]


    k = 0
    for thing in iterate_datafiles_maxmem(pathlist, mat_to_data):
        [print(x['data'].shape) for x in thing]
        print(k)
        k+=1 
        pass





if __name__ == '__main__':
    main()




