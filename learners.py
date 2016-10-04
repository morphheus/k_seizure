"""Different learners used for the project"""
import lasagne
import theano
import theano.tensor as T
import numpy as np
import time

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuralNet():
    """Basic class for a neural nework"""
    def __init__(self):
        """Subclass must implement the self.<thing> listed below
        input_var
        target_var
        loss_fct
        updates
        network
        batchsize
        predict
        predict_test
        loss
        test_loss
        """
        raise NotImplementedError()

    def train(self, x_train, y_train, x_test, y_test, num_epochs, train_batchsize, display=False, tally_errs=True, **kwargs):
        """Trains the neural network with the input/target pairs"""

        # Setup the actual function calls
        train_fct = theano.function([self.input_var, self.target_var], self.loss, updates=self.updates)
        test_fct = theano.function([self.input_var, self.target_var], [self.loss_test, self.acc_test])

        # Option to turn off error tallying, in case it takes up too much processing time
        if tally_errs:
            epoch_errors = np.empty(shape=(2,num_epochs))
        else:
            epoch_errors = None

        # Train the learner over the specified number of epochs
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(x_train, y_train, train_batchsize):
                inputs, targets = batch
                err = train_fct(inputs, targets)
                train_err += err
                train_batches += 1

            # And a full pass over the test data:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(x_test, y_test, train_batchsize):
                inputs, targets = batch
                err, acc = test_fct(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += len(targets)

            final_epoch_acc = test_acc / test_batches * 100
            logger.debug("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
            logger.debug("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            logger.debug("  test loss:    \t\t{:.6f}".format(test_err/test_batches))
            logger.debug("  test accuracy:\t\t{:.2f} %".format(final_epoch_acc))

            if tally_errs:
                epoch_errors[0,epoch] = train_err
                epoch_errors[1,epoch] = test_err

        return epoch_errors, final_epoch_acc

    def build_update_fct(self, learning_rate, update_fct=lasagne.updates.sgd, **kwargs):
        """Defines the update function for the neural net. Uses stochastic grad descent by default"""
        self.updates = update_fct(self.loss, self.params, learning_rate=learning_rate, **kwargs)

    def prepare_loss(self):
        """Builds attributes required for the initiation of loss functions"""
        self.predict = lasagne.layers.get_output(self.network)
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)

    def set_loss_binary_crossentropy(self):
        """Sets the loss function as the binary cross-entropy"""
        self.prepare_loss()

        #self.loss = lasagne.objectives.squared_error(self.predict, self.target_var).mean()
        self.loss = lasagne.objectives.binary_crossentropy(T.clip(self.predict, 0.001, 0.999), self.target_var).mean()

        self.predict_test = lasagne.layers.get_output(self.network, deterministic=True)
        self.loss_test = lasagne.objectives.binary_crossentropy(T.clip(self.predict_test, 0.001, 0.999), self.target_var).mean()
        self.acc_test = T.mean(T.eq(T.round(self.predict_test), self.target_var), dtype=theano.config.floatX)


class L2Convnet(NeuralNet):
    """Convolutional Neural Net for classification"""
    def __init__(self, batchsize, input_size):
        """Builds a feed-forward neural network.
        batchsize:     Size of training batches. Select None for variable batchsize
        input_size:    Length of input vector for 1 sample
        """
        self.input_var = T.matrix('inputs')
        self.target_var = T.ivector('targets')
        self.batchsize = batchsize

        # Build the chain of layers
        llist = []
        llist.append(lasagne.layers.InputLayer(shape=(batchsize, input_size), input_var=self.input_var))
        llist.append(lasagne.layers.DenseLayer(llist[-1], 13))
        llist.append(lasagne.layers.DenseLayer(llist[-1], 1))

        self.network = llist[-1]
        self.layer_count = len(llist)

    def train(self, *args, **kwargs):
        """Training function specific for this network"""
        logger.debug('Training L2Convnet')
        learning_rate = 0.005
        self.set_loss_binary_crossentropy()
        self.build_update_fct(learning_rate)
        return super().train(*args, **kwargs)




def iterate_minibatches(x,y, batchsize):
    """Iterates sequentially over the first axis of x and y numpy arrays in the specified batchsize"""
    k = 0
    if batchsize <= 0:
        raise ValueError('Batchsize cannot be zero or negative')

    while (k+batchsize) <= x.shape[0]:
        tmp = slice(k, k+batchsize)
        yield x[tmp, ...], y[tmp, ...]
        k += batchsize





if __name__ == '__main__':
    pass
    



    
