"""Contains all the basic learners used in this project"""
import numpy as np
import time
import logging
from logging import log

import learners
import datareader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    """Testing function"""
    np.random.seed(12121)
    feature_count = 3
    tot_samples = 200
    data = np.random.rand(tot_samples*feature_count).reshape(-1,feature_count).astype(np.float32)
    targets = np.random.randint(2, size=(tot_samples,)).astype(np.int32)
    data *= 0.8
    data[targets==1, 2] *= 0.2 
    data[targets==1, 2] += 0.8
    targets.reshape(-1,1)

    train_batchsize = 1
    num_epochs = 10

    learner_list = [(learners.L2Convnet(None, feature_count), (num_epochs, train_batchsize), {'display':False})]


    kfold_train(learner_list, data, targets)





def kfold_train(learner_list, data, targets, display=True):
    """Trains the list of learners in parallel.
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
        for x_train, y_train, x_test, y_test in datareader.stratified_folds(data, targets, folds=2):
            if display:
                logger.info('Working on <' + type(l[0]).__name__ + '> fold ' + str(k))
            err_lists, test_acc = l[0].train(x_train, y_train, x_test, y_test, *l[1], **l[2])
            all_errors.append(err_lists)
            accuracies.append(test_acc)
            k += 1

        print(len(all_errors))
        logger.debug(accuracies)






    #TODO: implement multiprocessing



if __name__ == '__main__':
    main()
