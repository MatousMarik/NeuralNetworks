import numpy as np


class Perceptron:
    """
    learning w.x := p == y -> pass
            else p == 0 -> w + lr*x
            else p == 1 -> w - lr*x
    => w + lr * x * (y - p)
        p == y => w + 0
        p == 0 => w + 1*...
        p == 1 => w - 1*...
    """

    def __init__(self, init_weights=[], lr=1, max_epochs=1000):
        # Perceptron constructor
        # self.weights will be initialized as
        # np.asarray(init_weights, dtype=float) - this enables that
        # init_weights can be either list of floats, or numpy array.
        # The empty weigth vector can be tested using np.any(self.weights).
        self.weights = np.asarray(init_weights, dtype=float)
        self.lr = lr
        self.max_epochs = max_epochs

    def predict(self, X):
        # Compute output of the perceptron.
        # Input X can be
        #  * vector, i.e., one sample
        #  * or a two-dimensional array, where each row is a sample.
        # Returns
        #  * either one value 0/1 if the input was a single vector
        #  * or vector with values 0/1 with the output of the perceptron
        #    for all samples in X
        # Raises an exception if the weights are not initialized.
        if self.weights.size == 0:
            raise Exception("Weights not initialized.")
        if X.ndim == 1:
            X_ = np.append(X, 1)
        else:
            X_ = np.column_stack([X, np.ones(X.shape[0])])
        return (np.dot(X_, self.weights) >= 0).astype(int)

    def partial_fit(self, X_, y, lr=1.0):
        # perform one epoch perceptron learning algorithm
        # on the training set (+ones) `X_` (two-dimensional numpy array of floats) with
        # the desired outputs `y` (vector of integers 0/1) and learning rate `lr`.
        # If self.weights is empty, the weight vector is generated randomly.
        if self.weights.size == 0:
            self.weights = np.random.uniform(-0.5, 0.5, X_.shape[-1])
        for x, g in zip(X_, y):
            self.weights += (g - (np.dot(x, self.weights) >= 0)) * lr * x

    def fit(self, X, y, lr=None, max_epochs=None):
        # trains perceptron using perceptron learning algorithm
        # on the training set `X` (two-dimensional numpy array of floats) with
        # the desired outputs `y` (vector of integers 0/1).
        # If self.weights is empty, the weight vector is generated randomly.
        # If the learning rate `lr == None`,
        # `self.lr` is used. If `max_epochs == None`, self.max_epochs is used.
        # Returns the number of epochs used in the training (at most `max_epochs`).
        if lr is None:
            lr = self.lr
        if max_epochs is None:
            max_epochs = self.max_epochs
        X_ = np.column_stack([X, np.ones(X.shape[0])])
        if self.weights is None:
            self.weights = np.random.uniform(-0.5, 0.5, X_.shape[-1])
        for i in range(max_epochs):
            if self.score(X, y) == 1:
                return i
            self.partial_fit(X_, y, lr)
        return max_epochs

    def score(self, X, y):
        # returns the mean accuracy on the test data (1.0 means that all prediction are correct,
        # 0.0 means that all predictions are wrong)
        # `X` (two-dimensional numpy array of floats) are test samples and
        # `y` (vector of integers 0/1) are correct labels for the test samples.
        return np.mean(self.predict(X) == y)
