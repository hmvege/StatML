#!/usr/bin/env python3
import numpy as np
import scipy.linalg
import numba as nb
import copy as cp

try:
    import utils.metrics as metrics
except ModuleNotFoundError:
    import metrics

__all__ = ["OLSRegression", "RidgeRegression", "LassoRegression"]


class __RegBackend:
    """Backend class in case we want to run with either scipy, numpy 
    (or something else)."""
    _fit_performed = False
    __possible_backends = ["numpy", "scipy"]
    __possible_inverse_methods = ["inv", "svd"]

    def __init__(self, linalg_backend="scipy", inverse_method="svd"):
        """Sets up the linalg backend."""
        assert linalg_backend in self.__possible_backends, \
            "{:s} backend not recognized".format(str(linalg_backend))
        self.linalg_backend = linalg_backend

        assert inverse_method in self.__possible_inverse_methods, \
            "{:s} inverse method not recognized".format(str(inverse_method))
        self.inverse_method = inverse_method

    def fit(self, X_train, y_train):
        raise NotImplementedError("Derived class missing fit()")

    def _check_if_fitted(self):
        """Small check if fit has been performed."""
        assert self._fit_performed, "Fit not performed"

    def score(self, X, y_true):
        """Returns the R^2 score.

        Args:
            X (ndarray): X array of shape (N, p - 1) to test for
            y_true (ndarray): true values for X

        Returns:
            float: R2 score for X_test values.
        """
        return metrics.r2(y_true, self.predict(X))

    def beta_variance(self):
        """Returns the variance of beta."""
        self._check_if_fitted()
        return self.coef_var

    def get_y_variance(self):
        if hasattr(self, "y_variance"):
            return self.y_variance
        else:
            raise AttributeError(
                ("Class {:s} does not contain "
                    "y_variance.".format(self.__class__)))

    @staticmethod
    @nb.njit(cache=True)
    def _predict(X, weights):
        return X @ weights

    def predict(self, X_test):
        """Performs a prediction for given beta coefs.

        Args:
            X_test (ndarray): test samples, size (N, p - 1)

        Returns:
            ndarray: test values for X_test
        """
        self._check_if_fitted()
        return self._predict(X_test, self.coef)

    def get_results(self):
        """Method for retrieving results from fit.

        Returns:
            y_approx (ndarray): y approximated on training data x.
            beta (ndarray):  the beta fit paramters.
            beta_cov (ndarray): covariance matrix of the beta values.
            beta_var (ndarray): variance of the beta values.
            eps (ndarray): the residues of y_train and y_approx.

        """
        return self.y_approx, self.coef, self.coef_cov, self.coef_var, self.eps

    @property
    def coef_(self):
        return self.coef

    @coef_.getter
    def coef_(self):
        return self.coef

    @coef_.setter
    def coef_(self, value):
        self.coef = value

    @property
    def coef_var(self):
        return self.beta_coefs_var

    @coef_var.getter
    def coef_var(self):
        return self.beta_coefs_var

    @coef_var.setter
    def coef_var(self, value):
        self.beta_coefs_var = value


class OLSRegression(__RegBackend):
    """
    An implementation of linear regression.

    Performs a fit on p features and s samples.
    """

    def __init__(self, **kwargs):
        """Initilizer for Linear Regression

        Args:
            linalg_backend (str): optional, default is "numpy". Choices: 
                numpy, scipy.
            inverse_method (str): optional, default is "svd". Choices:
                svd, inv.
        """
        super().__init__(**kwargs)

    @staticmethod
    @nb.njit(cache=True)
    def _ols_base(X_train, y_train):
        # N samples, P features
        N, P = X_train.shape

        # X^T * X
        XTX = X_train.T @ X_train

        # (X^T * X)^{-1}
        U, S, VH = np.linalg.svd(XTX)  # Using numpys svd method
        S = np.diag(1.0/S)

        XTX_inv = U @ S @ VH

        # Beta fit values: beta = (X^T * X)^{-1} @ X^T @ y
        coef = XTX_inv @ X_train.T @ y_train

        # y approximate. X @ beta
        y_approx = X_train @ coef

        # Residues.
        eps = y_train - y_approx

        # Variance of y approximate values. sigma^2, unbiased
        # y_variance = np.sum(eps**2) / float(N)
        y_variance = np.sum(eps**2) / (N - P - 1)

        # Beta fit covariance/variance. (X^T * X)^{-1} * sigma^2
        coef_cov = XTX_inv * y_variance
        coef_var = np.diag(coef_cov)

        return N, P, XTX, XTX_inv, coef, y_approx, y_approx, \
            eps, y_variance, coef_cov, coef_var

    def fit(self, X_train, y_train):
        """Fits/trains y_train with X_train using Linear Regression.

        X_train given as [1, x, x*2, ...]

        Args:
            X_train (ndarray): design matrix, (N, p - 1), 
            y_train (ndarray): (N),
        """
        self.N, self.P, _, _, self.coef, self.y_approx, self.y_approx, \
            self.eps, self.y_variance, self.coef_cov, self.coef_var = \
            self._ols_base(X_train, y_train)

        self._fit_performed = True


class RidgeRegression(__RegBackend):
    """
    An implementation of ridge regression.
    """

    def __init__(self, alpha=1.0, **kwargs):
        """A method for Ridge Regression.

        Args:
            alpha (float): alpha/lambda to use in Ridge Regression.
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    @staticmethod
    @nb.njit(cache=True)
    def _ridge_base(X_train, y_train, alpha):
        # N samples, P features
        N, P = X_train.shape

        # X^T * X
        XTX = X_train.T @ X_train

        # (X^T * X)^{-1}
        XTX_aI = XTX + alpha*np.eye(XTX.shape[0])

        # (X^T * X)^{-1}
        U, S, VH = np.linalg.svd(XTX_aI)  # Using numpys svd method
        S = np.diag(1.0/S)
        XTX_aI_inv = U @ S @ VH

        # Beta fit values: beta = (X^T * X)^{-1} @ X^T @ y
        coef = XTX_aI_inv @ X_train.T @ y_train

        # y approximate. X @ beta
        y_approx = X_train @ coef

        # Residues.
        eps = y_train - y_approx

        # Variance of y approximate values. sigma^2, unbiased
        # y_variance = metrics.mse(y_train, y_approx)
        y_variance = np.sum(eps**2) / (N - P - 1)

        # Beta fit covariance/variance.
        # See page 10 section 1.4 in https://arxiv.org/pdf/1509.09169.pdf
        # **REMEMBER TO CITE THIS/DERIVE THIS YOURSELF!**
        coef_cov = XTX_aI_inv @ XTX @ XTX_aI_inv.T
        coef_cov *= y_variance
        coef_var = np.diag(coef_cov)

        return N, P, XTX_aI, XTX_aI_inv, coef, y_approx, y_approx, \
            eps, y_variance, coef_cov, coef_var

    def fit(self, X_train, y_train):
        """Fits/trains y_train with X_train using Ridge Regression.

        X_train given as [1, x, x*2, ...]

        Args:
            X_train (ndarray): design matrix, (N, p - 1), 
            y_train (ndarray): (N, 1),
        """
        self.N, self.P, _, _, self.coef, self.y_approx, self.y_approx, \
            self.eps, self.y_variance, self.coef_cov, self.coef_var = \
            self._ridge_base(X_train, y_train, self.alpha)

        self._fit_performed = True


class LassoRegression(__RegBackend):
    """
    An implementation of lasso regression.
    """

    def __init__(self, alpha=1.0, **kwargs):
        """A method for Lasso Regression.

        Args:
            alpha (float): alpha/lambda to use in Lasso Regression.
        """
        super().__init__(**kwargs)
        self.alpha = alpha

        raise NotImplementedError

    def fit(self, X_train, y_train):
        raise NotImplementedError


def __test_ols_regression(x, y, deg):
    print("\nTesting OLS for  degree={}".format(deg))
    import sklearn.preprocessing as sk_preproc
    import sklearn.linear_model as sk_model

    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(x, y)

    reg = OLSRegression()
    reg.fit(X, y)
    print("Manual OLS regression")
    print("R^2: {:.16f}".format(reg.score(X, y)))

    sk_reg = sk_model.LinearRegression(fit_intercept=False, n_jobs=4)
    sk_reg.fit(X, y)
    print("SciKit OLS regression")
    print("R^2: {:.16f}".format(sk_reg.score(X, y)))

    c1, c2 = reg.coef_, sk_reg.coef_.T
    for i in range(reg.coef_.shape[0]):
        print("COEFF DIFF: {} - {} = {}".format(
            c1[i][0], c2[i][0], c1[i][0]-c2[i][0]))


def __test_ridge_regression(x, y, deg, alpha=1.0):
    print("\nTesting Ridge for degree={} for alpha={}".format(deg, alpha))
    import sklearn.preprocessing as sk_preproc
    import sklearn.linear_model as sk_model

    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(x, y)

    reg = RidgeRegression(alpha=alpha)
    reg.fit(X, y)
    print("Manual Ridge regression")
    print("R^2: {:.16f}".format(reg.score(X, y)))

    sk_reg = sk_model.Ridge(alpha=alpha, fit_intercept=False)
    sk_reg.fit(X, y)
    print("SciKit Ridge regression")
    print("R^2: {:.16f}".format(sk_reg.score(X, y)))


def __test_regresssions():
    n = 1000  # n cases, i = 0,1,2,...n-1
    deg = 10
    noise_strength = 0.1
    np.random.seed(1)
    x = np.random.rand(n, 1)
    y = 5.0*x*x + np.exp(-x*x) + noise_strength*np.random.randn(n, 1)

    __test_ols_regression(x, y, deg)

    for alpha_ in [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2]:
        __test_ridge_regression(x, y, deg, alpha_)
        __test_lasso_regression(x, y, deg, alpha_)


def __test_lasso_regression(x, y, deg, alpha=0.1):
    print("\nTesting Lasso for degree={} for alpha={}".format(deg, alpha))
    import sklearn.preprocessing as sk_preproc
    import sklearn.linear_model as sk_model

    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(x, y)

    sk_reg = sk_model.Lasso(alpha=alpha, fit_intercept=False)
    sk_reg.fit(X, y)
    print("SciKit Lasso regression")
    print("R^2: {:.16f}".format(sk_reg.score(X, y)))


if __name__ == '__main__':
    __test_regresssions()
