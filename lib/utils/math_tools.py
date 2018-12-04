#!/usr/bin/env python3

import numpy as np
import abc
from scipy.special import expit
import numba as nb

AVAILABLE_ACTIVATIONS = ["identity", "sigmoid", "relu", "tanh", "heaviside"]

AVAILABLE_OUTPUT_ACTIVATIONS = [
    "identity", "sigmoid", "softmax"]

AVAILABLE_COST_FUNCTIONS = ["mse", "log_loss", "exponential_cost",
                            "hellinger_distance",
                            "kullback_leibler_divergence",
                            "generalized_kullback_leibler_divergence",
                            "itakura_saito_distance"]

AVAILABLE_REGULARIZATIONS = ["l1", "l2", "elastic_net"]

# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

class _ActivationCore:
    @staticmethod
    @abc.abstractmethod
    def activate(x):
        pass

    @staticmethod
    @abc.abstractmethod
    def derivative(x):
        pass


class Sigmoid(_ActivationCore):
    @staticmethod
    @nb.njit(cache=True)
    def activate(x):
        """Sigmoidal activation function.

        Args:
            x (ndarray): weighted sum of inputs
        """
        # return expit(x)
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    @nb.njit(cache=True)
    def derivative(x):
        # s = Sigmoid.activate(x)
        s = 1.0/(1.0 + np.exp(-x))
        return s*(1-s)
    

class Identity(_ActivationCore):
    @staticmethod
    @nb.njit(cache=True)
    def activate(x):
        """Identity activation function. Input equals output.

        Args:
            x (ndarray): weighted sum of inputs.
        """
        return x

    @staticmethod
    @nb.njit(cache=True)
    def derivative(x):
        """Simply returns the derivative, 1, of the identity."""
        return np.ones(x.shape)


class Softmax(_ActivationCore):
    @staticmethod
    @nb.njit(cache=True)
    def activate(x):
        """The Softmax activation function. Assures that no outliers can 
        dominate too much.

        Numerically stable sosftmax
        https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

        Args:
            x (ndarray): weighted sum of inputs.
        """

        z_exp = np.exp((x - np.max(x)))
        return z_exp/np.sum(z_exp)

    @staticmethod
    def derivative(x):
        """The derivative of the Softmax activation function.

        Args:
            x (ndarray): weighted sum of inputs.
        """

        S = Softmax.activate(x).reshape(-1, 1)
        # return S - np.einsum("i...,j...->i...", S, S)
        return (np.identity(S.shape[0]) - np.dot(S, S.T)).sum(axis=1,
                                                              keepdims=True)

class SoftmaxCrossEntropy(_ActivationCore):
    @staticmethod
    @nb.njit(cache=True)
    def activate(x):
        """The Softmax activation function. Assures that no outliers can 
        dominate too much.

        Numerically stable sosftmax
        https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

        Args:
            x (ndarray): weighted sum of inputs.
        """

        z_exp = np.exp((x - np.max(x)))
        return z_exp/np.sum(z_exp)

    @staticmethod
    def derivative(x):
        """The derivative of the Softmax activation function.

        Args:
            x (ndarray): weighted sum of inputs.
        """
        return np.empty((1))


class Heaviside(_ActivationCore):
    @staticmethod
    def activate(x):
        """The Heaviside activation function.

        Args:
            x (ndarray): weighted sum of inputs
        """
        return np.where(x >= 0, 1, 0)

    @staticmethod
    @nb.njit(cache=True)
    def derivative(x):
        """The derivative of the Heaviside activation function.

        Args:
            x (ndarray): weighted sum of inputs
        """
        return np.zeros(x.shape)


class Relu(_ActivationCore):
    @staticmethod
    def activate(x):
        """The rectifier activation function. Only activates if argument x is 
        positive.

        Args:
            x (ndarray): weighted sum of inputs
        """
        # np.clip(x, 0, np.finfo(x.dtype).max, out=x)
        # return x
        return np.where(x >= 0, x, 0)

    @staticmethod
    def derivative(x):
        """The derivative of the tangens hyperbolicus activation function.

        Args:
            x (ndarray): weighted sum of inputs
        """
        return np.where(Relu.activate(x) > 0, 1, 0)


class Tanh(_ActivationCore):
    @staticmethod
    @nb.njit(cache=True)
    def activate(x):
        """The tangens hyperbolicus activation function.

        Args:
            x (ndarray): weighted sum of inputs
        """
        return np.tanh(x)

    @staticmethod
    @nb.njit(cache=True)
    def derivative(x):
        """The derivative of the tangens hyperbolicus activation function.

        Args:
            x (ndarray): weighted sum of inputs
        """
        return 1 - np.tanh(x)**2


# =============================================================================
# COST FUNCTIONS
# =============================================================================

class _BaseCost:
    """Base cost function class."""
    @staticmethod
    @abc.abstractmethod
    def cost(a, y):
        """Returns cost function.

        Args:
            a (ndarray): layer output.
            y (ndarray): true output.
        Return:
            (float): cost function output.
        """
        return None

    @staticmethod
    @abc.abstractmethod
    def delta(a, y, x):
        return None


class MSECost(_BaseCost):
    @staticmethod
    @nb.njit(cache=True)
    def cost(a, y):
        """Returns cost function.

        Args:
            a (ndarray): all layer outputs.
            y (ndarray): all true outputs.
        Return:
            (float): cost function output.
        """
        return 0.5*np.linalg.norm(a - y)**2 / float(a.shape[0])
        # return 0.5*np.mean(np.linalg.norm(a - y, axis=1)**2, axis=0)

    @staticmethod
    @nb.njit(cache=True)
    def delta(a, y, a_prime):
        return (a - y) * a_prime


class LogEntropyCost(_BaseCost):
    """
    Cross entropy cost function.

    Only to be used with softmax output layer.
    """
    @staticmethod
    @nb.njit(cache=True)
    def cost(a, y):
        """Returns cost function.

        Args:
            a (ndarray): layer output.
            y (ndarray): true output.
        Return:
            (float): cost function output.
        """
        return - np.mean(y*np.log(a))  # + (1 - y)*np.log(1 - a))

    @staticmethod
    @nb.njit(cache=True)
    def delta(a, y, a_prime):
        return a - y  # For softmax
        # return (- y / a) * a_prime # General expr, still a bit bugged smh


class ExponentialCost(_BaseCost):
    """Exponential cost function."""
    @staticmethod
    # @nb.njit(cache=True)
    def cost(a, y, tau=0.1):
        """Returns cost function.

        Args:
            a (ndarray): layer output.
            y (ndarray): true output.
        Return:
            (float): cost function output.
        """
        return tau*np.exp(1/tau * np.sum((y-y_true)**2))

    @staticmethod
    # @nb.njit(cache=True)
    def delta(a, y, a_prime, tau=0.1):
        """Exponential cost function gradient."""
        return 2/tau * (y-y_true)*self.cost(y, y_true, tau) * a_prime


# =============================================================================
# REGULARIZATIONS
# =============================================================================
class _BaseRegularization:
    """Base cost function class."""
    @staticmethod
    @abc.abstractmethod
    def __call__(weights):
        """Returns the regularization."""
        return None

    @staticmethod
    @abc.abstractmethod
    def delta(weights):
        return None


class L1Regularization(_BaseRegularization):
    """The L1 norm."""
    @staticmethod
    def __call__(weights):
        """Returns the regularization."""
        return np.linalg.norm(weights, ord=1)

    @staticmethod
    @nb.njit(cache=True)
    def derivative(weights):
        """The derivative of the L1 norm."""
        # NOTE: Include this in report
        # https://math.stackexchange.com/questions/141101/
        # minimizing-l-1-regularization
        return np.sign(weights)


class L2Regularization(_BaseRegularization):
    """The L2 norm."""
    @staticmethod
    def __call__(weights):
        """Returns the regularization."""
        return 0.5*np.dot(weights, weights)

    @staticmethod
    @nb.njit(cache=True)
    def derivative(weights):
        """The derivative of the L2 norm."""
        # NOTE: Include this in report
        # https://math.stackexchange.com/questions/2792390/derivative-of-
        # euclidean-norm-l2-norm
        return weights


class ElasticNetRegularization(_BaseRegularization):
    """The elastic net regularization, L_en = L1 + L2."""
    @staticmethod
    def __call__(weights):
        """Returns the regularization."""
        return np.linalg.norm(weights, ord=1) + 0.5*np.dot(weights, weights)

    @staticmethod
    @nb.njit(cache=True)
    def derivative(weights):
        """
        Derivative of elastic net is just L1 and L2 derivatives combined.
        """
        return np.sign(weights) + weights
