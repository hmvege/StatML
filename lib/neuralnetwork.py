import numpy as np
import copy as cp
from tqdm import tqdm, trange
try:
    import utils.math_tools as umath
    from utils.math_tools import AVAILABLE_ACTIVATIONS, \
        AVAILABLE_OUTPUT_ACTIVATIONS, AVAILABLE_COST_FUNCTIONS, \
        AVAILABLE_REGULARIZATIONS
except ModuleNotFoundError:
    import lib.utils.math_tools as umath
    from lib.utils.math_tools import AVAILABLE_ACTIVATIONS, \
        AVAILABLE_OUTPUT_ACTIVATIONS, AVAILABLE_COST_FUNCTIONS, \
        AVAILABLE_REGULARIZATIONS

# Temporary imports for performance testing
import time
import numba as nb

# TODO: implement the use of all cost functions for any output activation.
# TODO: vectorize mini batch procedure.
# TODO: parallelize the mini batch procedure.
# TODO: create yml file for conda env

def plot_image(sample_, label, pred):
    """Simple function for plotting the input."""
    sample = cp.deepcopy(sample_)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    plt.imshow(
        sample.reshape(int(np.sqrt(sample.shape[0])),
                       int(np.sqrt(sample.shape[0]))),
        cmap=cm.gray)
    title_str = "Label: {} Prediction: {}".format(label, pred)
    print(title_str)
    plt.title(title_str)
    plt.show()


class MultilayerPerceptron:
    def __init__(self, layer_sizes, activation="sigmoid",
                 output_activation="sigmoid", cost_function="mse", alpha=0.0,
                 regularization="l2", momentum=0.0, weight_init="default"):
        """Initializer for multilayer perceptron.

        Number of layers is always minimum N_layers + 2.

        Args:
            layer_sizes (list(int)): list of layer sizes after input data.
                Constists of [input_layer_size, N layer sizes, output_layer].
            activation (str): activation function. Choices is "sigmoid", 
                "identity", "relu", "tanh", "heaviside". Optional, default is 
                "sigmoid".
            output_activation (str): final layer activation function. Choices 
                is "sigmoid" or "sigmoid", "softmax", "identity". Optional, 
                default is "sigmoid".
            cost_function (str): Cost function. Choices is "mse", "log_loss". 
                Optional, default "mse".
            alpha (float): L2 regularization term. Default is 0.0.
            regularization (str): Regularization type. Choices: l1, l2, 
                elastic_net.
            momentum (float): adds a dependency on previous gradient.
            weight_init (str): weight initialization. Choices: 'large', 
                'default'. Large weight sets initial weights to a gaussian 
                distribution with sigma=1 and mean=0. Default sets to
                sigma=1/sqrt(N_samples) and mean=0.
        Raises:
            AssertionError: if input_data_size is not a list.
            AssertionError: if layer_sizes is less than two.
        """
        assert isinstance(layer_sizes, list), "must provide a layer size list"
        assert len(layer_sizes) >= 2, ("Must have at least two layers: "
                                       "len(layer_sizes)={}".format(
                                           len(layer_sizes)))

        self._set_layer_activation(activation)
        self._set_output_layer_activation(output_activation)
        self._set_cost_function(cost_function)
        self._set_regularization(regularization)

        # L2 regularization term
        self.alpha = alpha

        # For storing epoch evaluation scores
        self.epoch_evaluations = []

        # Sets momentum given it is valid
        assert momentum >= 0.0, "momentum must be positive"
        self.momentum = momentum

        # Sets up weights
        if weight_init == "large":
            # l_i, l_j is the layer input-output sizes.
            self.weights = [
                np.random.randn(l_j, l_i)
                for l_i, l_j in zip(layer_sizes[:-1], layer_sizes[1:])]
        else:
            self.weights = [
                np.random.randn(l_j, l_i) / np.sqrt(l_i)
                for l_i, l_j in zip(layer_sizes[:-1], layer_sizes[1:])]

        # Sets up biases
        self.biases = [np.random.randn(l_j, 1) for l_j in layer_sizes[1:]]

        # Sets up activations and forward-pass values
        self.activations = [np.empty(s_).reshape(-1, 1) for s_ in layer_sizes]
        self.z = [np.empty(b.shape) for b in self.biases]

        self.layer_sizes = layer_sizes
        self.N_layers = len(layer_sizes)

    def _set_layer_activation(self, activation):
        """Sets the layer activation."""
        assert activation in AVAILABLE_ACTIVATIONS, (
            "{} not among available output activation functions: "
            "{}".format(activation, ", ".join(AVAILABLE_ACTIVATIONS)))

        self.activation = activation

        if activation == "sigmoid":
            self._activation = umath.Sigmoid
        elif activation == "identity":
            self._activation = umath.Identity
        elif activation == "relu":
            self._activation = umath.Relu
        elif activation == "tanh":
            self._activation = umath.Tanh
        elif activation == "heaviside":
            self._activation = umath.Heaviside
        else:
            raise KeyError("Activation type '{}' not recognized. Available "
                           "activations:".format(
                               activation, ", ".join(AVAILABLE_ACTIVATIONS)))

    def _set_output_layer_activation(self, output_activation):
        """Sets the final layer activation."""

        assert output_activation in AVAILABLE_OUTPUT_ACTIVATIONS, (
            "{} not among available output activation functions: "
            "{}".format(output_activation, ", ".join(
                AVAILABLE_OUTPUT_ACTIVATIONS)))

        self.output_activation = output_activation

        if output_activation == "sigmoid":
            self._output_activation = umath.Sigmoid
        elif output_activation == "identity":
            self._output_activation = umath.Identity
        elif output_activation == "softmax":
            self._output_activation = umath.Softmax
        else:
            raise KeyError("Final layer activation type '{}' not "
                           "recognized. Available activations:".format(
                               activation, ", ".join(
                                   AVAILABLE_OUTPUT_ACTIVATIONS)))

    def _set_learning_rate(self, eta, eta0=1.0):
        """Sets the learning rate."""
        if isinstance(eta, float):
            self._update_learning_rate = lambda _i, _N: eta
        elif eta == "inverse":
            self._update_learning_rate = lambda _i, _N: eta0 * \
                (1 - _i/float(_N+1))
        else:
            raise KeyError(("Eta {} is not recognized learning"
                            " rate.".format(eta)))

    def _set_cost_function(self, cost_function):
        """Sets the cost function to use.

        A nice list of different cost functions found here:
        https://stats.stackexchange.com/questions/154879/a-list-of-
        cost-functions-used-in-neural-networks-alongside-applications

        Args:
            cost_functions (str): name of the cost function to use.

        Raises:
            KeyError if cost_function is not a recognized cost function.
        """
        self.cost_function = cost_function
        if cost_function == "mse":
            self._cost = umath.MSECost()
        elif cost_function == "log_loss":
            assert self.output_activation == "softmax", (
                "Only softmax output activation can be used with "
                "log_loss(Cross Entropy) cost function. "
                "Provided output activation: {}".format(
                    self.output_activation))
            self._output_activation = umath.SoftmaxCrossEntropy()
            self._cost = umath.LogEntropyCost()
        elif cost_function == "exponential_cost":
            self._cost = umath.ExponentialCost()
        elif cost_function == "hellinger_distance":
            raise NotImplementedError(cost_function)
        elif cost_function == "kullback_leibler_divergence":
            raise NotImplementedError(cost_function)
        elif cost_function == "generalized_kullback_leibler_divergence":
            raise NotImplementedError(cost_function)
        elif cost_function == "itakura_saito_distance":
            raise NotImplementedError(cost_function)
        else:
            raise KeyError("Cost function '{}' not recognized. Available loss"
                           " functions: {}".format(cost_function, ", ".join(
                               AVAILABLE_COST_FUNCTIONS)))

    def _set_regularization(self, regularization):
        """Sets the regularization."""
        self.regularization = regularization
        if regularization == "l1":
            self._regularization = umath.L1Regularization()
        elif regularization == "l2":
            self._regularization = umath.L2Regularization()
        elif regularization == "elastic_net":
            self._regularization = umath.ElasticNetRegularization()
        else:
            raise KeyError("Regularization {} not recognized. Available"
                           " regularizations: {}".format(
                               regularization,
                               ", ".join(AVAILABLE_REGULARIZATIONS)))

    def _get_reg(self, layer):
        """Computes the L2 regularization.

        Args:
            layer (int): layer to compute regularization for.

        Returns:
            (float) l2-norm of given layer.
        """
        if self.alpha != 0.0:
            return self.alpha * \
                self._regularization(self.weights[layer])
        else:
            return 0.0

    def _get_reg_delta(self, layer):
        """Computes the L2 regularization derivative.

        Args:
            layer (int): layer to compute regularization for.

        Returns:
            (ndarray) derivative of the l2-norm of given layer.
        """
        if self.alpha != 0.0:
            return self.alpha * \
                self._regularization.derivative(self.weights[layer])
        else:
            return 0.0

    def predict(self, x):
        """Returns the last layer of activation from _forward_pass."""
        return self._forward_pass(x)[-1]

    def _forward_pass(self, activation):
        """Performs a feed-forward to the last layer."""
        activations = [activation]
        for i in range(self.N_layers - 1):
            z = (self.weights[i] @ activations[i])
            z += self.biases[i]

            if i+1 != (self.N_layers - 1):
                # activations.append(self._activation.activate(self._fp_core(
                #     self.weights[i], self.biases[i], activations[i])))
                activations.append(self._activation.activate(z))
            # activations.append(self._activation.activate(z))

        # activations.append(self._output_activation.activate(self._fp_core(
        #             self.weights[-1], self.biases[-1], activations[-1])))
        activations.append(self._output_activation.activate(z))

        return activations

    def _back_propagate(self, x, y):
        """Performs back-propagation on a single dataset.

        Args:
            x (ndarray): initial layer input.
            y (ndarray): true output values(labels), one-hot vector.

        Returns:
            (list(ndarray)): all layer weight gradients
            (list(ndarray)): all layer bias gradients
        """

        # Retrieves the z and sigmoid for each layer in sample
        self.activations[0] = x

        for i in range(self.N_layers - 1):
            self.z[i] = self.weights[i] @ self.activations[i]
            self.z[i] += self.biases[i]

            if (i+1) != (self.N_layers - 1):
                # Middle layer(s) activation
                self.activations[i+1] = self._activation.activate(self.z[i])
            else:
                # Sigmoid output layer
                self.activations[i+1] = \
                    self._output_activation.activate(self.z[i])

        # Backpropegation begins, initializes the backpropagation derivatives
        delta_w = [np.empty(w.shape) for w in self.weights]
        delta_b = [np.empty(b.shape) for b in self.biases]

        # Gets initial delta value, first of the four equations
        delta = self._cost.delta(
            self.activations[-1].T, y,
            self._output_activation.derivative(self.z[-1]).T).T

        # Sets last element before back-propagating
        delta_b[-1] = delta
        delta_w[-1] = delta @ self.activations[-2].T
        # delta_w[-1] = np.einsum("ijk,ilk->ijl", delta, self.activations[-2].T)

        delta_w[-1] += self._get_reg_delta(-1)  # /x.shape[0]

        # Loops over layers
        for l in range(2, self.N_layers):
            # Retrieves the z and gets it's derivative
            z_ = self.z[-l]
            sp = self._activation.derivative(z_)

            # Sets up delta^l
            delta = self.weights[-l+1].T @ delta
            delta *= sp

            delta_b[-l] = delta  # np.sum(delta, axis=1)
            delta_w[-l] = delta @ self.activations[-l-1].T
            # delta_w[-l] = np.einsum("ijk,ilk->ijl", delta, self.activations[-l-1].T)

            delta_w[-l] += self._get_reg_delta(-l)

        return delta_w, delta_b

    def train(self, data_train, data_train_labels, epochs=10,
              mini_batch_size=50, eta=1.0, eta0=1.0,
              data_test=None, data_test_labels=None, verbose=False):
        """Trains the neural-net on provided data. Assumes data size 
        is the same as what provided in the initialization.

        Uses Stochastic Gradient Descent(SGA) and mini-batches to get the 
        deed done.

        Args:
            data_train (ndarray): training data. Shape: 
                (samples, input_size, 1)
            data_train_labels (ndarray): training data labels. Shape: 
                (samples, output_size)
            epochs (int): number of times we are to train the data. Default 
                is 10.
            mini_batch_size (int): size of mini batch. Optional, default is 50.
            eta (float): learning rate, optional. Choices: float(constant), 
                'inverse'. "Inverse" sets eta to 1 - i/(N+1). Default is 1.0.
            eta0 (float): learning rate start for 'inverse'.
            data_test (ndarray): data to run tests for. Shape:
                (samples, input_size, 1)
            data_test_labels (ndarray): training data labels. Shape: 
                (samples, output_size)
            verbose (bool): if prompted, will print cost and evaluation scores.

        Raises:
            AssertionError: if input data to not match the specified layer 
                data given in the initialization.
        """

        assert self.layer_sizes[0] == data_train.shape[1], (
            "training data "
            "and labels do not match in shape: {} != {}".format(
                self.layer_sizes[0], data_train.shape[1]))

        # Sets if we are to evaluate the data while running
        if (not isinstance(data_test, type(None))) and \
                (not isinstance(data_test_labels, type(None))):
            perform_eval = True
        else:
            perform_eval = False

        N_train_size = data_train.shape[0]

        # Gets the number of batches
        number_batches = N_train_size // mini_batch_size

        self._set_learning_rate(eta, eta0)

        for epoch in trange(epochs, desc="Epoch"):

            # if epoch==0:
            #     tqdm.write("Cost: {}".format(
            #             self.cost(data_train, data_train_labels)))

            # Updates the learning rate
            eta_ = self._update_learning_rate(epoch, epochs)

            # Performs the SGA step of shuffling data
            shuffle_indexes = np.random.choice(list(range(N_train_size)),
                                               size=N_train_size,
                                               replace=False)

            # Shuffles the data with the shuffle-indices
            shuffled_data = data_train[shuffle_indexes]
            shuffled_labels = data_train_labels[shuffle_indexes]

            # Splits data into minibatches
            shuffled_data = [
                shuffled_data[i:i+mini_batch_size]
                for i in range(0, N_train_size, number_batches)]
            shuffled_labels = [
                shuffled_labels[i:i+mini_batch_size]
                for i in range(0, N_train_size, number_batches)]

            # avg_update_time = []

            # Iterates over mini batches
            for mb_data, mb_labels in zip(shuffled_data, shuffled_labels):

                # # Time tracking
                # t0 = time.time()

                self.update_mini_batch(mb_data, mb_labels, eta_)

                # # Time tracking
                # t1 = time.time()
                # avg_update_time.append(t1-t0)

            # # Time tracking
            # tqdm.write("Mean MB update time: {}".format(
            #     np.mean(avg_update_time)))

            # If we have provided testing data, we perform an epoch evaluation
            if perform_eval:
                self.epoch_evaluations.append(
                    np.sum(self.evaluate(data_test, data_test_labels)))

                if verbose:
                    tqdm.write("Cost: {}".format(
                        self.cost(data_train, data_train_labels)))

                    tqdm.write("Score: {}/{}".format(
                        self.epoch_evaluations[-1],
                        len(data_test_labels)))

    def update_mini_batch(self, mb_data, mb_labels, eta):
        """Trains the network on the mini batch."""

        # delta_w, delta_b = self._back_propagate(mb_data, mb_labels)

        # for l in range(self.N_layers - 1):
        #     self.weights[l] -= np.mean(delta_w[l], axis=0)*eta
        #     self.biases[l] -= np.mean(delta_b[l], axis=0)*eta

        # Resets gradient sums
        delta_w_sum = [np.zeros(w.shape) for w in self.weights]
        delta_b_sum = [np.zeros(b.shape) for b in self.biases]

        # Loops over all samples and labels in mini batch
        for sample, label in zip(mb_data, mb_labels):

            # Runs back-propagation
            delta_w, delta_b = self._back_propagate(sample, label)

            # Sums the derivatives into a single list of derivative-arrays.
            delta_w_sum = [dws + dw for dw, dws in zip(delta_w, delta_w_sum)]
            delta_b_sum = [dbs + db for db, dbs in zip(delta_b, delta_b_sum)]

        # Updates weights and biases by subtracting their gradients
        for l in range(self.N_layers - 1):
            self.weights[l] -= (delta_w_sum[l]*eta/len(mb_data))
            self.biases[l] -= (delta_b_sum[l]*eta/len(mb_data))

    def cost(self, X, y):
        """Calculates the cost."""
        y_pred = np.asarray([self.predict(ix) for ix in X]).reshape(*y.shape)
        return self._cost.cost(y_pred, y)

    def evaluate(self, test_data, test_labels, show_image=False):
        """Evaluates test data.

        Args:
            test_data (ndarray): array of shape (sample, input_size, 1), 
                contains the input data to test for.
            test_labels (ndarray): array of desired output to compare against.
                On the shape of (sample, output_size)
            show_image (bool): plots input values. Assumes input is square. 
                Optional, default is False.
        """

        results = []
        for test, label in zip(test_data, test_labels):
            pred = self.predict(np.atleast_2d(test))
            prediction_bool = int(np.argmax(pred) == np.argmax(label))
            results.append(prediction_bool)

            if show_image or not prediction_bool:
                plot_image(test, np.argmax(label), np.argmax(pred))

        return results

    def score(self, test_data, test_labels, verbose=False):
        """Returns the accuracy score for given test data.

        Args:
            test_data (ndarray): array of shape (sample, input_size, 1), 
                contains the input data to test for.
            test_labels (ndarray): array of desired output to compare against.
                On the shape of (sample, output_size)        
        """
        results = self.evaluate(test_data, test_labels)
        if verbose:
            print("Accuracy = {}/{} = {}".format(np.sum(results), len(results),
                                                 np.mean(results)))
        return np.mean(results)


def __test_mlp_mnist():
    import pickle

    test_data_path = "../datafiles/HandwritingClassification/mnist.pkl"
    with open(test_data_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        data_train, data_valid, data_test = u.load()

    print("DATA TRAIN: ", data_train[0].shape, data_train[1].shape)
    print("DATA VALID: ", data_valid[0].shape, data_valid[1].shape)
    print("DATA TEST: ", data_test[0].shape, data_test[1].shape)

    def convert_output(label_, output_size):
        """Converts label to output vector."""
        y_ = np.zeros(output_size, dtype=float)
        y_[label_] = 1.0
        return y_

    # Converts data to ((N, p-1)) shape.
    data_train_samples = np.asarray(
        [d_.reshape((-1, 1)) for d_ in data_train[0]])
    data_valid_samples = np.asarray(
        [d_.reshape((-1, 1)) for d_ in data_valid[0]])
    data_test_samples = np.asarray(
        [d_.reshape((-1, 1)) for d_ in data_test[0]])

    # Converts labels from single floats to arrays with 1.0 at correct output.
    # Aka, to one-hot vector format.
    data_train_labels = np.asarray(
        [convert_output(l, 10) for l in data_train[1]])
    data_valid_labels = np.asarray(
        [convert_output(l, 10) for l in data_valid[1]])
    data_test_labels = np.asarray(
        [convert_output(l, 10) for l in data_test[1]])

    ################################################
    # Parameters
    ################################################
    # Activation options: "sigmoid", "identity", "relu", "tanh", "heaviside"
    activation = "tanh"
    # Cost function options: "mse", "log_loss", "exponential_cost"
    cost_function = "mse"
    # Output activation options:  "identity", "sigmoid", "softmax"
    output_activation = "sigmoid"
    # Weight initialization options:
    # default(sigma=1/sqrt(N_samples)), large(sigma=1.0)
    weight_init = "default"
    alpha = 0.0
    regularization = "l2"
    mini_batch_size = 20
    epochs = 100
    eta = "inverse"  # Options: float, 'inverse'
    eta0 = 1.0
    verbose = True

    # Sets up my MLP.
    MLP = MultilayerPerceptron([data_train_samples.shape[1], 30, 10],
                               activation=activation,
                               cost_function=cost_function,
                               output_activation=output_activation,
                               weight_init=weight_init,
                               alpha=alpha,
                               regularization=regularization)
    # Timing
    t0_train = time.time()

    MLP.train(data_train_samples, data_train_labels,
              data_test=data_test_samples,
              data_test_labels=data_test_labels,
              mini_batch_size=mini_batch_size,
              epochs=epochs,
              eta=eta,
              eta0=eta0,
              verbose=verbose)

    # Timing
    t1_train = time.time()
    print("*"*100)
    print("Training time: {0:.4f} seconds".format(t1_train-t0_train))
    print("*"*100)

    print(MLP.score(data_test_samples, data_test_labels))
    MLP.evaluate(data_test_samples, data_test_labels, show_image=False)


def __test_nn_sklearn_comparison():
    import warnings
    import copy as cp
    from sklearn.neural_network import MLPRegressor

    def test_regressor(X_train, y_train, X_test, y_test, nn_layers,
                       sk_hidden_layers, input_activation, output_activation,
                       alpha=0.0):

        if input_activation == "sigmoid":
            sk_input_activation = "logistic"
        else:
            sk_input_activation = input_activation

        if output_activation == "sigmoid":
            sk_output_activation = "logistic"
        else:
            sk_output_activation = output_activation

        mlp = MLPRegressor(
            solver='sgd',               # Stochastic gradient descent.
            activation=sk_input_activation,  # Skl name for sigmoid.
            alpha=alpha,                  # No regularization for simplicity.
            hidden_layer_sizes=sk_hidden_layers)  # Full NN size is (1,3,3,1).

        mlp.out_activation_ = sk_output_activation

        # Force sklearn to set up all the necessary matrices by fitting a data
        # set. We dont care if it converges or not, so lets ignore raised
        # warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_train, y_train)

        # =====================================================================
        n_samples, n_features = X_train.shape
        batch_size = n_samples
        hidden_layer_sizes = mlp.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        layer_units = ([n_features] + hidden_layer_sizes + [mlp.n_outputs_])
        activations = [X_test]
        activations.extend(np.empty((batch_size, n_fan_out))
                           for n_fan_out in layer_units[1:])
        deltas = [np.empty_like(a_layer) for a_layer in activations]
        coef_grads = [np.empty((n_fan_in_, n_fan_out_))
                      for n_fan_in_, n_fan_out_ in zip(layer_units[:-1],
                                                       layer_units[1:])]
        intercept_grads = [np.empty(n_fan_out_)
                           for n_fan_out_ in layer_units[1:]]
        # =====================================================================

        mlp.out_activation_ = sk_output_activation
        activations = mlp._forward_pass(activations)
        loss, coef_grads, intercept_grads = mlp._backprop(
            X_test, y_test, activations, deltas, coef_grads, intercept_grads)

        # Activates my own MLP
        nn = MultilayerPerceptron(
            nn_layers, 
            activation=input_activation,
            output_activation=output_activation, 
            cost_function="mse",
            alpha=alpha)

        # Copy the weights and biases from the scikit-learn network to your
        # own.
        for i, w in enumerate(mlp.coefs_):
            nn.weights[i] = cp.deepcopy(w.T)
        for i, b in enumerate(mlp.intercepts_):
            nn.biases[i] = cp.deepcopy(b.T.reshape(-1, 1))

        # Call your own backpropagation function, and you're ready to compare
        # with the scikit-learn code.
        y_sklearn = mlp.predict(X_test)
        y = nn.predict(cp.deepcopy(X_test).T)

        # Asserts that the forward pass is correct
        assert np.allclose(y, y_sklearn), (
            "Prediction {} != {}".format(y, y_sklearn))

        delta_w, delta_b = nn._back_propagate(X_test.T, y_test)

        # Asserts that the the activations is correct in back propagation
        for i, a in enumerate(nn.activations):
            print(i, a.T, activations[i])
            assert np.allclose(
                a.T, activations[i]), "error in layer {}".format(i)
        else:
            print("Activations are correct.")

        # Asserts that the the biases is correct in back propagation
        for i, derivative_bias in enumerate(delta_b):
            print(i, derivative_bias.T, intercept_grads[i])
            assert np.allclose(
                derivative_bias.T, intercept_grads[i]), (
                "error in layer {}".format(i))
        else:
            print("Biases derivatives are correct.")

        # Asserts that the the weights is correct in back propagation
        for i, derivative_weight in enumerate(delta_w):
            print(i, derivative_weight.T, coef_grads[i])
            assert np.allclose(derivative_weight.T,
                               coef_grads[i]), "error in layer {}".format(i)
        else:
            print("Weight derivatives are correct.")

        print("Test complete\n")

    # Training data
    X_train1 = np.array([[0.0], [1.0]])
    y_train1 = np.array([0, 2])
    layer_sizes1 = [1, 3, 3, 1]
    sk_hidden_layers1 = (3, 3)

    X_train2 = np.array([[0.0, 0.5], [1.0, 1.5]])
    y_train2 = np.array([0, 1.0])
    layer_sizes2 = [2, 3, 3, 2]
    sk_hidden_layers2 = (3, 3)

    X_train3 = np.random.rand(100, 10)
    y_train3 = np.random.rand(100)
    layer_sizes3 = [10, 20, 20, 10]
    sk_hidden_layers3 = (20, 20)

    # Completely random data point(s) which we will propagate through
    # the network.
    X_test1 = np.array([[1.125982598]])
    y_test1 = np.array([8.29289285])

    X_test2 = np.array([[1.125982598, 2.937172838]])
    y_test2 = np.array([8.29289285])

    X_test3 = np.array([np.random.rand(10)])
    y_test3 = np.array([8.29289285])

    # Note: need to run with (a-y), and not (a-y)*a_prime in delta derivative.
    test_regressor(X_train1, y_train1, X_test1, y_test1,
                   layer_sizes1, sk_hidden_layers1, "sigmoid", "softmax")

    test_regressor(X_train2, y_train2, X_test2, y_test2,
                   layer_sizes2, sk_hidden_layers2, "sigmoid", "identity",
                   alpha=0.5)

    test_regressor(X_train3, y_train3, X_test3, y_test3,
                   layer_sizes3, sk_hidden_layers3, "sigmoid", "sigmoid")

    print("Forward and back propagation tests passed.\n")


if __name__ == '__main__':
    __test_mlp_mnist()
    # __test_nn_sklearn_comparison()
