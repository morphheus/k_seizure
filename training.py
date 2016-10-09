"""Contains all the basic learners used in this project"""
import numpy as np
import time
import logging
import warnings
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from logging import log

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=UserWarning)
    import theano

import learners
import datareader
import plotlib as graphs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def kfold_train(learner_list, data, targets, folds, show_hair=True, display=True):
    """Trains the list of learners in parallel. PARALLEL NOT IMPLEMENTED YET
    All learners must be objects having the following function signature:
    <learner>.train(x_train, y_train, x_test, y_test, num_epochs, train_batchsize, display=False)

    learner_list: list of tuples: (learner_obj, args, kwargs)

    """
    if type(learner_list) != type(list()):
        raise ValueError('Expected a list of learners as a first argument, even for single learners')

    for l in learner_list:
        all_errors = []
        accuracies = []
        k = 0
        tf = time.clock
        t0 = tf()
        for x_train, y_train, x_test, y_test in stratified_folds(data, targets, folds=folds):
            if display:
                if k!= 0:
                    logger.info('Fold ' + str(k-1) + ': done in ' + "%2.2f"%(tf()-t0) + ' sec')
                logger.info('Working on <' + type(l[0]).__name__ + '> fold ' + str(k))
                t0 = tf()
            # If x/ydata is a string, assume it is a path and load it
            epochs_errors, test_acc = l[0].train(x_train, y_train, x_test, y_test, *l[1], **l[2])
            all_errors.append(epochs_errors)
            accuracies.append(test_acc)
            k += 1

        logger.debug(accuracies)
        if show_hair: graphs.train_and_test(all_errors, legend_labels=('Training folds', 'Testing folds')); graphs.show()


def stratified_folds(X, Y, folds=7, shuffle_split=False):
    """Returns the indicies for training folds and test sets. Note that the shuffle option corresponds to a ShuffleSplit cross-validator""" 
    sss = StratifiedShuffleSplit(n_splits=folds, test_size=0.2) if shuffle_split else StratifiedKFold(n_splits=folds, shuffle=False)
    for train_index, test_index in sss.split(X,Y):
        list_of_index = lambda x, l: [x[k] for k in l]
        x_train, y_train = [list_of_index(var, train_index) for var in (X,Y)]
        x_test, y_test = [list_of_index(var, test_index) for var in (X,Y)]
        yield x_train, y_train ,x_test, y_test





def main():
    """Testing function"""

    train_batchsize = 1
    num_epochs = 10


    datapaths, targets = datareader.get_data_paths(1, train=True, preproc='base')
    #x = datareader.mat_to_data(datapaths[0])
    #print(x['data'].nbytes); 
    #print(x['data'].shape); exit()
    x = np.load(datapaths[0])
    feature_count = x.shape[0]

    
    learner_list = [(learners.L2Convnet(None, feature_count), (num_epochs, train_batchsize), {'display':False})]

    kfold_train(learner_list, datapaths, targets, 3)




    #TODO: implement multiprocessing



if __name__ == '__main__':
    main()
