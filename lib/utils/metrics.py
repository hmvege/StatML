#!/usr/bin/env python
import numpy as np
import numba as nb

__all__ = ["mse", "r2", "bias", "timing_function"]


# TODO: jit what can be jitted


def mse(y_exact, y_predict, axis=None):
    """Mean Square Error

    Uses numpy to calculate the mean square error.

    MSE = (1/n) sum^(N-1)_i=0 (y_i - y_test_i)^2

    Args:
        y_exact (ndarray): response/outcome/dependent variable,
            size (N_samples, 1)
        y_predict (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: mean square error
    """

    assert y_exact.shape == y_predict.shape, ("y_exact.shape = {} y_predict"
        ".shape = {}".format(y_exact.shape, y_predict.shape))

    return np.mean((y_exact - y_predict)**2, axis=axis)

def r2(y_exact, y_predict, axis=None):
    """R^2 score

    Uses numpy to calculate the R^2 score.

    R^2 = 1 - sum(y - y_test)/sum(y - mean(y_test))

    Args:
        y_exact (ndarray): response/outcome/dependent variable,
            size (N_samples, 1)
        y_predict (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: R^2 score
    """

    mse_exact_pred = np.sum((y_exact - y_predict)**2, axis=axis)
    variance_exact = np.sum((y_exact - np.mean(y_exact))**2)
    return (1.0 - mse_exact_pred/variance_exact)


def bias(y_exact, y_predict, axis=0):
    """Bias^2 of a exact y and a predicted y

    Args:
        y_exact (ndarray): response/outcome/dependent variable,
            size (N_samples, 1)
        y_predict (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: Bias^2
    """
    return np.mean((y_predict - np.mean(y_exact, keepdims=True, axis=axis))**2)


# def ridge_regression_variance(X, sigma2, lmb):
#     """Analytical variance for beta coefs in Ridge regression,
#     from section 1.4.2, page 10, https://arxiv.org/pdf/1509.09169.pdf"""
#     XT_X = X.T @ X
#     W_lmb = XT_X + lmb * np.eye(XT_X.shape[0])
#     W_lmb_inv = np.linalg.inv(W_lmb)
#     return np.diag(sigma2 * W_lmb_inv @ XT_X @ W_lmb_inv.T)


def timing_function(func):
    """Time function decorator."""
    import time

    def wrapper(*args, **kwargs):
        t1 = time.clock()

        val = func(*args, **kwargs)

        t2 = time.clock()

        time_used = t2-t1

        print("Time used with function {:s}: {:.10f} secs/ "
              "{:.10f} minutes".format(
                func.__name__, time_used, time_used/60.))

        return val

    return wrapper
