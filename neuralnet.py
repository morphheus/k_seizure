"""Neural network methods"""
import lasagne
import theano
import theano.tensor as T
import numpy as np
import time

class NeuralNet():
    """Basic class for a neural nework"""
    def __init__(self):
        """Subclass must implement self.
        input_var
        target_var
        loss_fct
        updates
        network
        batchsize
        loss
        """
        raise NotImplementedError()

    def train(self, x_train, y_train, num_epochs, **kwargs):
        """Trains the neural network with the input/target pairs"""
        train_fct = theano.function([self.input_var, self.target_var], self.loss, updates=self.updates)

        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(x_train, y_train, self.batchsize):
                inputs, targets = batch
                train_err += train_fct(inputs, targets)
                train_batches += 1

            ## And a full pass over the validation data:
            #val_err = 0
            #val_acc = 0
            #val_batches = 0
            #for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            #    inputs, targets = batch
            #    err, acc = val_fn(inputs, targets)
            #    val_err += err
            #    val_acc += acc
            #    val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            #print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            #print("  validation accuracy:\t\t{:.2f} %".format(
            #    val_acc / val_batches * 100))

    def build_update_fct(self, learning_rate, update_fct=lasagne.updates.sgd, **kwargs):
        """Defines the update function for the neural net. Uses stochastic grad descent by default"""
        self.updates = update_fct(self.loss, self.params, learning_rate=learning_rate, **kwargs)

class DFFNN(NeuralNet):
    """Dense feed forward neural network"""
    def __init__(self, batchsize, units_list, **hidden_kwargs):
        """Builds a feed-forward neural network.
        batchsize:     Size of training batches
        units_list:    List of integers representing the number of units per layer. Must be at
                       least of length 2 to create an input/output layer
        hidden_kwargs: Kwargs to be passed to the hidden layer constructor
        """
        if len(units_list) < 2:
            raise ValueError('Expected an iterable of length 2')

        self.input_var = T.matrix('inputs')
        self.target_var = T.matrix('targets')
        self.batchsize = batchsize

        # Build the chain of layers
        llist = []
        llist.append(lasagne.layers.InputLayer(shape=(batchsize, units_list[0]), input_var=self.input_var))
        for k in units_list[1:-1]:
            llist.append(lasagne.layers.DenseLayer(llist[-1], k, **hidden_kwargs))
        llist.append(lasagne.layers.DenseLayer(llist[-1], units_list[-1]))

        self.network = llist[-1]
        self.layer_count = len(llist)

        self.predict = lasagne.layers.get_output(self.network)
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)

    def loss_mean_err(self):
        self.loss = ((self.predict - self.target_var)**2).mean()






def iterate_minibatches(x,y, batchsize):
    """Iterates sequentially over the first axis of x and y numpy arrays in the specified batchsize"""
    if x.shape != y.shape:
        raise Exception('x and y must have the same shape')
    if x.shape[0] % batchsize:
        raise Exception('The number of inputs is not an integer multiple of the batchsize')

    k = 0
    while (k+batchsize) <= x.shape[0]:
        tmp = slice(k, k+batchsize)
        yield x[tmp, ...], y[tmp, ...]
        k += batchsize





if __name__ == '__main__':
    data = np.eye(10, dtype=np.float32)
    batchsize = 10

    neuralnet = DFFNN(batchsize, [10,3,10])
    neuralnet.loss_mean_err()
    neuralnet.build_update_fct(1)
    neuralnet.train(data, data, 100)
    



    
