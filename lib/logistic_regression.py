#!/usr/bin/env python3

import numpy as np
import scipy
import copy as cp
import warnings
try:
    import utils.math_tools as umath
    import utils.optimize as uopt
    from utils.math_tools import AVAILABLE_OUTPUT_ACTIVATIONS
except ModuleNotFoundError:
    import lib.utils.math_tools as umath
    import lib.utils.optimize as uopt
    from lib.utils.math_tools import AVAILABLE_OUTPUT_ACTIVATIONS


# TODO: validate SGD without minibatches
# TODO: validate SGD with minibatches
# TODO: implement momentum
# TODO: clean up cost function - move outside?
# TODO: parallized logreg
# TODO: make optimizers take penalty argument?


class LogisticRegression:
    """An implementation of Logistic regression."""
    _fit_performed = False

    def __init__(self, solver="lr-gd", activation="sigmoid",
                 max_iter=100, penalty="l2", tol=1e-8, alpha=1.0,
                 momentum=0.0, mini_batch_size=50):
        """Sets up the linalg backend.

        Args:
            solver (str): what kind of solver method to use. Default is 
                'lr-gd' (gradient descent). Choices: 'lr-gd', 'gd', 'cg', 
                'sga', 'sga-mb', 'nr', 'newton-cg'.
            activation (str): type of activation function to use. Optional, 
                default is 'sigmoid'.
            max_iter (int): number of iterations to run gradient descent for,
                default is 100.
            penalty (str): what kind of regulizer to use, either 'l1' or 'l2'. 
                Optional, default is 'l2'.
            tol (float): tolerance or when to cut of calculations. Optional, 
                default is 1e-4.
            alpha (float): regularization strength. Default is 1.0.
            momentum (float): adds a momentum, in which the current gradient 
                deepends on the last gradient. Default is 0.0.
            mini_batch_size (int): size of mini-batches. Only available for 
                sga-mb. Optional, default is 50.
        """

        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.momentum = momentum
        self.mini_batch_size = mini_batch_size

        self._set_optimizer(solver)
        self._set_activation_function(activation)
        self._set_regularization_method(penalty)

    def _set_optimizer(self, solver_method):
        """Set the penalty/regularization method to use."""
        self.solver_method = solver_method

        if solver_method == "lr-gd":
            # Steepest descent for Logistic Regression
            self.solver = uopt.LogRegGradientDescent(momentum=self.momentum)
        elif solver_method == "gd":
            # aka Steepest descent
            self.solver = uopt.GradientDescent(momentum=self.momentum)
        elif solver_method == "cg":
            # Conjugate gradient method
            self._chech_momentum(solver_method)
            self.solver = uopt.ConjugateGradient()
        elif solver_method == "sga":
            # Stochastic Gradient Descent
            self.solver = uopt.SGA(momentum=self.momentum)
        elif solver_method == "sga-mb":
            # Stochastic Gradient Descent with mini batches
            self.solver = uopt.SGA(momentum=self.momentum,
                                   use_minibatches=True,
                                   mini_batch_size=self.mini_batch_size)
        elif solver_method == "nr":
            # Newton-Raphson method
            self._chech_momentum(solver_method)
            self.solver = uopt.NewtonRaphson()
        elif solver_method == "newton-cg":
            # Newton-Raphson method
            self._chech_momentum(solver_method)
            self.solver = uopt.NewtonCG()
        else:
            raise KeyError(("{} not recognized as a solver"
                            " method. Choices: {}.".format(
                                solver_method,
                                ", ".join(uopt.OPTIMIZERS_KEYWORDS))))

    def _chech_momentum(self, solver_method):
        """Raises error for given solver method if momentum is nonzero, 
        as solver method do not have momentum capabilities."""
        if self.momentum != 0:
            raise ValueError("Momentum not available for "
                "method {}".format(solver_method))

    def _set_regularization_method(self, penalty):
        """Set the penalty/regularization method to use."""
        self.penalty_type = penalty

        if penalty == "l1":
            self.penalty = umath.L1Regularization
            # self._get_penalty_derivative = umath._l1_derivative
        elif penalty == "l2":
            self.penalty = umath.L2Regularization
            # self._get_penalty_derivative = umath._l2_derivative
        elif penalty == "elastic_net":
            self.penalty = umath.ElasticNetRegularization
            # self._get_penalty_derivative = umath._elastic_net_derivative
        elif isinstance(type(penalty), None):
            self.penalty = umath.NoRegularization
        else:
            raise KeyError(("{} not recognized as a regularization"
                            " method.".format(penalty)))

    def _set_activation_function(self, activation):
        """Sets the final layer activation."""

        assert activation in AVAILABLE_OUTPUT_ACTIVATIONS, (
            "{} not among available output activation functions: "
            "{}".format(activation, ", ".join(
                AVAILABLE_OUTPUT_ACTIVATIONS)))

        self.activation = activation

        if activation == "sigmoid":
            self._activation = umath.sigmoid
        elif activation == "softmax":
            self._activation = umath.softmax
        else:
            raise KeyError("Final layer activation type '{}' not "
                           "recognized. Available activations:".format(
                               activation, ", ".join(
                                   AVAILABLE_OUTPUT_ACTIVATIONS)))

    @property
    def coef_(self):
        return self.coef

    @coef_.getter
    def coef_(self):
        return cp.deepcopy(self.coef)

    @coef_.setter
    def coef_(self, value):
        self.coef = value

    def fit(self, X_train, y_train, eta=1.0):
        """Performs a linear regression fit for data X_train and y_train.

        Args:
            X_train (ndarray): input data.
            y_train (ndarray): output one-hot labeled data.
            eta (float): learning rate, optional. Choices: float(constant), 
                "inverse". "Inverse" sets eta to 1 - i/(N+1). Default is 1.0.
        """
        X = cp.deepcopy(X_train)
        y = cp.deepcopy(y_train)

        self.N_features, self.p = X.shape
        assert y.shape[0] == self.N_features

        # Adds constant term and increments the number of predictors
        X = np.hstack([np.ones((self.N_features, 1)), X])

        # Adds beta_0 coefficients
        self.coef = np.zeros(self.p + 1)
        self.coef[0] = 1

        self.coef = self.solver.solve(X, y, self.coef, self._cost_function,
                                      self._cost_function_gradient, eta=0.01,
                                      max_iter=100000, tol=1e-6,
                                      scale=self.N_features,
                                      alpha=self.alpha)

        self._fit_performed = True

    def _cost_function(self, X, y, weights, eps=1e-15):
        """Cost/loss function for logistic regression. Also known as the 
        cross entropy in statistics.

        Args:
            X (ndarray): design matrix, shape (N, p).
            y (ndarray): predicted values, shape (N, labels).
            weights (ndarray): matrix of coefficients (p, labels).
        Returns:
            (ndarray): 1D array of predictions
        """

        p_ = np.dot(X, weights)
        loss = - np.sum(y*p_ - np.log(1 + np.exp(p_)))
        loss += (0.5*self.alpha*np.dot(weights, weights))
        return loss

        # y_pred = self._predict(X, weights)

        # p_probabilities = self._activation(y_pred)

        # # Removes bad values and replaces them with limiting values eps
        # p_probabilities = np.clip(p_probabilities, eps, 1-eps)

        # cost1 = - y * np.log(p_probabilities)
        # cost2 = (1 - y) * np.log(1 - p_probabilities)
        # cost = np.sum(cost1 - cost2) + self._get_penalty(weights)*self.alpha

        # return cost

    def _cost_function_gradient(self, X, y, weights):
        """Takes the gradient of the cost function w.r.t. the coefficients.

            dC(W)/dw = - X^T * (y - p(X^T * w))
        """

        # p_ = umath.sigmoid(X @ weights)
        # loss = - X.T @ (y - p_) + self.alpha * weights
        # return loss

        grad = np.dot(X.T, (self._activation(self._predict(X, weights)) - y))

        # Adds regularization
        grad += self.alpha*weights

        return grad

    def _cost_function_laplacian(self, X, y, w):
        """Takes the laplacian of the cost function w.r.t. the coefficients.

            d^2C(w) / (w w^T) = X^T W X
        where
            W = p(1 - X^T * w) * p(X^T * w)
        """
        y_pred = self._predict(X, w)
        return X.T @ self._activation(1-y_pred) @ self._activation(y_pred) @ X

    def _predict(self, X, weights):
        """Performs a regular fit of parameters and sends them through 
        the sigmoid function.

        Args:
            X (ndarray): design matrix/feature matrix, shape (N, p)
            weights (ndarray): coefficients 
        """
        return X @ weights

    def score(self, X, y):
        """Returns the mean accuracy of the fit.

        Args:
            X (ndarray): array of shape (N, p - 1) to classify.
            Y (ndarray): true labels.

        Returns:
            (float): mean accuracy score for features_test values.
        """
        pred = self.predict(X)
        accuracies = np.sum(self._indicator(pred, y))

        return accuracies/float(y.shape[0])

    def _indicator(self, features_test, labels_test):
        """Returns 1 if features_test[i] == labels_test[i]

        Args:
            features_test (ndarray): array of shape (N, p - 1) to test for
            labels_test (ndarray): true labels

        Returns:
            (array): elements are 1 or 0
        """
        return np.where(features_test == labels_test, 1, 0)

    def predict(self, X):
        """Predicts category 1 or 2 of X.

        Args:
            X (ndarray): design matrix of shape (N, p - 1)
        """
        if not self._fit_performed:
            raise UserWarning("Fit not performed.")

        # Adds intercept
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Retrieves probabilitites
        probabilities = self._activation(self._predict(X, self.coef)).ravel()

        # Sets up binary probability
        results_proba = np.asarray([1 - probabilities, probabilities])

        # Moves axis from (2, N_probabilitites) to (N_probabilitites, 2)
        results_proba = np.moveaxis(results_proba, 0, 1)

        # Sets up binary prediction of either 0 or one
        results = np.where(results_proba[:, 0] >= results_proba[:, 1], 0, 1).T

        return results

    def predict_proba(self, X):
        """Predicts probability of a design matrix X of shape (N, p - 1)."""
        if not self._fit_performed:
            raise UserWarning("Fit not performed.")

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        probabilities = self._activation(self._predict(X, self.coef)).ravel()
        results = np.asarray([1 - probabilities, probabilities])

        return np.moveaxis(results, 0, 1)


def __test_logistic_regression():
    from sklearn import datasets
    import sklearn.linear_model as sk_model
    import sklearn.model_selection as sk_modsel
    import matplotlib.pyplot as plt

    iris = datasets.load_iris()
    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

    # Local implementation parameters
    test_size = 0.1
    penalty = "elastic_net"
    learning_rate = 0.001
    max_iter = 1000000
    # Available solvers:
    # ["lr-gd", "gd", "cg", "sga", "sga-mb", "nr", "newton-cg"]
    solver = "lr-gd"
    # solver = "newton-cg"
    activation = "sigmoid"
    tol = 1e-8
    alpha = 0.1
    momentum = 0.0
    mini_batch_size = 20

    # Sets up test and training data
    X_train, X_test, y_train, y_test = \
        sk_modsel.train_test_split(X, y, test_size=test_size, shuffle=True)
    X_new = np.linspace(0, 3, 100).reshape(-1, 1)

    # Manual logistic regression
    print ("Manual solver method:", solver)
    log_reg = LogisticRegression(penalty=penalty, solver=solver,
                                 activation=activation, tol=tol,
                                 alpha=alpha, momentum=momentum,
                                 mini_batch_size=mini_batch_size,
                                 max_iter=max_iter)
    log_reg.fit(cp.deepcopy(X_train), cp.deepcopy(
        y_train), eta=learning_rate)
    y_proba = log_reg.predict_proba(X_new)

    print("Manual log-reg coefs:", log_reg.coef_)

    # SK-Learn logistic regression
    if penalty == "elastic_net":
        sk_penalty = "l2"
    else: 
        sk_penalty = penalty
    sk_log_reg = sk_model.LogisticRegression(fit_intercept=True,
                                             C=1.0/alpha, penalty=sk_penalty,
                                             max_iter=max_iter, tol=tol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Removes annoing future-warning
        sk_log_reg.fit(cp.deepcopy(X_train), cp.deepcopy(y_train))
    y_sk_proba = sk_log_reg.predict_proba(X_new)

    print("SK-learn log-reg coefs: ", sk_log_reg.intercept_, sk_log_reg.coef_)

    # Sets coef with SK learns coef's for comparing outputs
    manual_coefs = log_reg.coef_
    sk_coefs = np.asarray(
        [sk_log_reg.intercept_[0], sk_log_reg.coef_[0, 0]])

    # =========================================================================
    # Runs tests with SK learn's coefficients, and checks that our
    # implementation's predictions match SK-learn's predictions.
    # =========================================================================

    print("Score before using SK-learn's coefficients: {0:.16f}".format(
        log_reg.score(X_test, y_test)))

    # Sets the coefficients from the SK-Learn to local method
    log_reg.coef_ = sk_coefs

    print("Score after using SK-learn's coefficients: {0:.16f}".format(
        log_reg.score(X_test, y_test)))

    # Asserts that predicted probabilities matches.
    y_sk_proba_compare = sk_log_reg.predict_proba(X_test)
    y_proba_compare = log_reg.predict_proba(X_test)
    assert np.allclose(y_sk_proba_compare, y_proba_compare), (
        "Predicted probabilities do not match: (SKLearn) {} != {} "
        "(local implementation)".format(y_sk_proba_compare, y_proba_compare))

    # Asserts that the labels match
    sk_predict = sk_log_reg.predict(X_test)
    local_predict = log_reg.predict(X_test)
    assert np.allclose(sk_predict, local_predict), (
        "Predicted class labels do not match: (SKLearn) {} != {} "
        "(local implementation)".format(sk_predict, local_predict))

    # Assert that the scores match
    sk_score = sk_log_reg.score(X_test, y_test)
    local_score = log_reg.score(X_test, y_test)
    assert np.allclose(sk_score, local_score), (
        "Predicted score do not match: (SKLearn) {} != {} "
        "(local implementation)".format(sk_score, local_score))

    fig1 = plt.figure()

    # SK-Learn logistic regression
    ax1 = fig1.add_subplot(211)
    ax1.plot(X_new, y_sk_proba[:, 1], "g-", label="Iris-Virginica(SK-Learn)")
    ax1.plot(X_new, y_sk_proba[:, 0], "b--",
             label="Not Iris-Virginica(SK-Learn)")
    ax1.set_title(
        r"SK-Learn versus manual implementation of Logistic Regression")
    ax1.set_ylabel(r"Probability")
    ax1.legend()

    # Manual logistic regression
    ax2 = fig1.add_subplot(212)
    ax2.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica(Manual)")
    ax2.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica(Manual)")
    ax2.set_ylabel(r"Probability")
    ax2.legend()

    # Plots decision boundary
    log_reg.coef_ = manual_coefs

    # Retrieves decision boundaries
    p_false_manual, p_true_manual = log_reg.predict_proba(X_new).T
    p_false_sk, p_true_sk = sk_log_reg.predict_proba(X_new).T

    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111)
    ax3.plot(X_train, y_train, "o")

    ax3.plot(X_new, p_true_manual, label="Manual true")
    ax3.plot(X_new, p_false_manual, label="Manual false")
    ax3.plot(X_new, p_true_sk, label="SK-Learn true")
    ax3.plot(X_new, p_false_sk, label="SK-Learn false")
    ax3.legend()
    ax3.axhline(0.5)
    ax3.axvline(X_new[int(len(X_new)/2.0)])
    ax3.set_title("Decision boundary")

    # plt.show()


if __name__ == '__main__':
    __test_logistic_regression()
