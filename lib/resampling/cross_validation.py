#!/usr/bin/env python3
import numpy as np
import copy as cp
from tqdm import tqdm

try:
    import sys
    sys.path.insert(0, "../")
    from utils import metrics
except ModuleNotFoundError:
    import metrics

import sklearn.model_selection as sk_modsel
import sklearn.metrics as sk_metrics

import numba as nb
import multiprocessing as mp

__all__ = ["kFoldCrossValidation", "MCCrossValidation"]

# TODO: parallelize kf-CV
# TODO: parallelize kkf-CV
# TODO: parallelize mc-kf-CV


class __CV_core:
    """Core class for performing k-fold cross validation."""
    _reg = None

    def __init__(self, X_data, y_data, reg, numprocs=1):
        """Initializer for Cross Validation.

        Args:
            X_data (ndarray): Design matrix on the shape (N, p)
            y_data (ndarray): y data on the shape (N, 1). Data to be 
                approximated.
            reg (Regression Instance): an initialized regression method
        """
        assert X_data.shape[0] == len(y_data), (
            "x and y data not of equal lengths")

        assert hasattr(reg, "fit"), ("regression method must have "
                                     "attribute fit()")
        assert hasattr(reg, "predict"), ("regression method must have "
                                         "attribute predict()")

        self.X_data = cp.deepcopy(X_data)
        self.y_data = cp.deepcopy(y_data)

        self._reg = reg

        if numprocs <= 1:
            self.numprocs = 1
        else:
            self.numprocs = numprocs

    @property
    def reg(self):
        return self._reg

    @reg.setter
    def reg(self, reg):
        """Args:
            rmethod (regression class): regression class to use
        """
        self._reg = reg

    @property
    def coef_(self):
        return self.coef_coefs

    @coef_.getter
    def coef_(self):
        return self.beta_coefs

    @property
    def coef_var(self):
        return self.beta_coefs_var

    @coef_var.getter
    def coef_var(self):
        return self.beta_coefs_var


class kFoldCrossValidation(__CV_core):
    """Class for performing k-fold cross validation."""

    @staticmethod
    # @nb.jit(cache=True)
    def _kfcv_core(input_values):
        """Core part of the kf-CV."""
        ik, set_list, X_subdata, y_subdata, X_test, reg = input_values
        # Sets up new data set
        k_X_train = np.concatenate([X_subdata[d] for d in set_list])
        k_y_train = np.concatenate([y_subdata[d] for d in set_list])

        # Trains method bu fitting data
        reg.fit(k_X_train, k_y_train)

        # Getting a prediction given the test data
        y_predict = reg.predict(X_test).ravel()

        # Returns prediction and beta coefs
        return ik, y_predict, reg.coef_

    def cross_validate(self, k_splits=5, test_percent=0.2, shuffle=False,
                       X_test=None, y_test=None):
        """
        Args:
            k_splits (float): percentage of the data which is to be used
                for cross validation. Default is 5.
            test_percent (float): size of test data in percent. Optional, 
                default is 0.2.
            X_test (ndarray): design matrix for test values, shape (N, p),
                optional.
            y_test (ndarray): y test data on shape (N, 1), optional.
        """

        N_total_size = self.X_data.shape[0]

        # Checks if we have provided test data or not
        if isinstance(X_test, type(None)) and \
                isinstance(y_test, type(None)):

            # Splits X data and design matrix data
            X_train, X_test, y_train, y_test = \
                sk_modsel.train_test_split(self.X_data, self.y_data,
                                           test_size=test_percent,
                                           shuffle=False)

        else:
            # If X_test and y_test is provided, we simply use those as test
            # values.
            X_train = self.X_data
            y_train = self.y_data
            X_test = cp.deepcopy(X_test)
            y_test = cp.deepcopy(y_test)

        test_size = y_test.shape[0]

        # Splits kfold train data into k actual folds
        X_subdata = np.array_split(X_train, k_splits, axis=0)
        y_subdata = np.array_split(y_train, k_splits, axis=0)

        # Stores the test values from each k trained data set in an array
        r2_list = np.empty(k_splits)
        beta_coefs = np.empty((k_splits, y_train.shape[-1], X_test.shape[-1]))
        self.y_pred_list = np.empty((test_size, k_splits))

        # Sets up set lists
        set_lists = []
        for ik in range(k_splits):
            tmp_set_list = list(range(k_splits))
            tmp_set_list.pop(ik)
            set_lists.append(tmp_set_list)

        if self.numprocs == 1:
            # Main kf-CV loop
            for ik in tqdm(range(k_splits), desc="k-fold Cross Validation"):

                # Sets up new data set
                k_X_train = np.concatenate(
                    [X_subdata[d] for d in set_lists[ik]])
                k_y_train = np.concatenate(
                    [y_subdata[d] for d in set_lists[ik]])

                # Trains method bu fitting data
                self.reg.fit(k_X_train, k_y_train)

                # Getting a prediction given the test data
                y_predict = self.reg.predict(X_test).ravel()

                # Appends prediction and beta coefs
                self.y_pred_list[:, ik] = y_predict
                beta_coefs = self.reg.coef_
        else:
            # Initializes multiprocessing
            with mp.Pool(processes=self.numprocs) as p:

                # Sets up jobs for parallel processing
                input_values = list(zip(list(range(k_splits)),
                                        set_lists,
                                        [X_subdata for i in range(k_splits)],
                                        [y_subdata for i in range(k_splits)],
                                        [X_test for i in range(k_splits)],
                                        [self.reg for i in range(k_splits)]))


                # Runs parallel processes
                # import time
                # t0 = time.time()
                results = p.imap_unordered(self._kfcv_core, input_values)
                # t1 = time.time()
                # print("TIME imap_unordered:", t1-t0)
                # for i_kf, res_ in enumerate(sorted(results, key=lambda k: k[0])):
                for i_kf, res_ in enumerate(results):
                    self.y_pred_list[:, i_kf] = res_[1]
                    beta_coefs[i_kf] = res_[2].T
                # exit(1)
                # return ik, y_predict, reg.coef_
                # # Populates result lists
                # for i_bs, res_ in enumerate(results):

        # Mean Square Error, mean((y - y_approx)**2)
        _mse = (y_test - self.y_pred_list)**2
        self.mse = np.mean(np.mean(_mse, axis=1, keepdims=True))

        # Bias, (y - mean(y_approx))^2
        _mean_pred = np.mean(self.y_pred_list, axis=1, keepdims=True)
        _bias = y_test - _mean_pred
        self.bias = np.mean(_bias**2)

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        # _r2 = metrics.r2(y_test, self.y_pred_list, axis=1)
        # self.r2 = np.mean(_r2)
        self.r2 = sk_metrics.r2_score(y_test, self.y_pred_list.mean(axis=1))

        # Variance, var(y_predictions)
        self.var = np.mean(np.var(self.y_pred_list, axis=1, keepdims=True))

        beta_coefs = np.asarray(beta_coefs)
        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=1)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=1)

        self.x_pred_test = X_test[:, 1]
        self.y_pred = np.mean(self.y_pred_list, axis=1)
        self.y_pred_var = np.var(self.y_pred_list, axis=1)


class kkFoldCrossValidation(__CV_core):
    """A nested k fold CV for getting bias."""

    def cross_validate(self, k_splits=4, test_percent=0.2, X_test=None,
                       y_test=None, shuffle=False):
        """
        Args:
            k_splits (float): Number of k folds to make in the data. Optional,
                default is 4 folds.
            test_percent (float): Percentage of data set to set aside for 
                testing. Optional, default is 0.2.
            X_test (ndarray): design matrix test data, shape (N,p). Optional, 
                default is using 0.2 percent of data as test data.
            y_test (ndarray): design matrix test data. Optional, default is 
                default is using 0.2 percent of data as test data.
        """

        # Checks if we have provided test data or not
        if isinstance(X_test, type(None)) and \
                isinstance(y_test, type(None)):

            # Splits X data and design matrix data
            X_train, X_test, y_train, y_test = \
                sk_modsel.train_test_split(self.X_data, self.y_data,
                                           test_size=test_percent,
                                           shuffle=shuffle)

        else:
            # If X_test and y_test is provided, we simply use those as test
            # values.
            X_train = self.X_data
            y_train = self.y_data
            X_test = cp.deepcopy(X_test)
            y_test = cp.deepcopy(y_test)

        N_total_size = X_train.shape[0]

        # Splits dataset into a holdout test chuck to find bias, variance ect
        # on and one to perform k-fold CV on.
        holdout_test_size = N_total_size // k_splits

        # In case we have an uneven split
        if (N_total_size % k_splits != 0):
            X_train = X_train[:holdout_test_size*k_splits]
            y_train = y_train[:holdout_test_size*k_splits]

        # Splits data
        X_data = np.split(X_train, k_splits, axis=0)
        y_data = np.split(y_train, k_splits, axis=0)

        # Sets up some arrays for storing the different MSE, bias, var, R^2
        # scores.
        mse_arr = np.empty(k_splits)
        r2_arr = np.empty(k_splits)
        var_arr = np.empty(k_splits)
        bias_arr = np.empty(k_splits)

        beta_coefs = []
        x_pred_test = []
        y_pred_mean_list = []
        y_pred_var_list = []

        for i_holdout in tqdm(range(k_splits),
                              desc="Nested k fold Cross Validation"):

            # Gets the testing holdout data to be used. Makes sure to use
            # every holdout test data once.
            X_holdout = X_data[i_holdout]
            y_holdout = y_data[i_holdout]

            # Sets up indexes
            holdout_set_list = list(range(k_splits))
            holdout_set_list.pop(i_holdout)

            # Sets up new holdout data sets
            X_holdout_train = np.concatenate(
                [X_data[d] for d in holdout_set_list])
            y_holdout_train = np.concatenate(
                [y_data[d] for d in holdout_set_list])

            # Splits dataset into managable k fold tests
            test_size = X_holdout_train.shape[0] // k_splits

            # Splits kfold train data into k actual folds
            X_subdata = np.array_split(X_holdout_train, k_splits, axis=0)
            y_subdata = np.array_split(y_holdout_train, k_splits, axis=0)

            # Stores the test values from each k trained data set in an array
            r2_list = np.empty(k_splits)

            y_pred_list = np.empty((X_test.shape[0], k_splits))

            # Loops over all k-k folds, ensuring every fold is used as a
            # holdout set.
            for ik in range(k_splits):

                # Sets up indexes
                set_list = list(range(k_splits))
                set_list.pop(ik)

                # Sets up new data set
                k_X_train = np.concatenate([X_subdata[d] for d in set_list])
                k_y_train = np.concatenate([y_subdata[d] for d in set_list])

                # Trains method bu fitting data
                self.reg.fit(k_X_train, k_y_train)

                # Appends prediction and beta coefs
                y_pred_list[:, ik] = self.reg.predict(X_test).ravel()
                beta_coefs.append(self.reg.coef_)

            # Mean Square Error, mean((y - y_approx)**2)
            _mse = (y_test - y_pred_list)**2
            mse_arr[i_holdout] = np.mean(np.mean(_mse, axis=1, keepdims=True))

            # Bias, (y - mean(y_approx))^2
            _mean_pred = np.mean(y_pred_list, axis=1, keepdims=True)
            _bias = y_test - _mean_pred
            bias_arr[i_holdout] = np.mean(_bias**2)

            # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
            r2_arr[i_holdout] = metrics.r2(
                y_test, y_pred_list.mean(axis=1, keepdims=True))

            # Variance, var(y_predictions)
            _var = np.var(y_pred_list, axis=1, keepdims=True)
            var_arr[i_holdout] = np.mean(_var)

            y_pred_mean_list.append(np.mean(y_pred_list, axis=1))
            y_pred_var_list.append(np.var(y_pred_list, axis=1))

        self.var = np.mean(var_arr)
        self.bias = np.mean(bias_arr)
        self.r2 = np.mean(r2_arr)
        # self.r2 = sk_metrics.r2_score(y_test, y_predict.mean(axis=1))
        self.mse = np.mean(mse_arr)
        beta_coefs = np.asarray(beta_coefs)
        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=0)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=0)

        self.x_pred_test = X_test[:, 1]
        self.y_pred = np.array(y_pred_mean_list).mean(axis=0)
        self.y_pred_var = np.array(y_pred_var_list).mean(axis=0)


class MCCrossValidation(__CV_core):
    """
    https://stats.stackexchange.com/questions/51416/k-fold-vs-monte-carlo-cross-validation
    """

    def cross_validate(self, N_mc, k_splits=4, test_percent=0.2, X_test=None,
                       y_test=None, shuffle=False):
        """
        Args:
            N_mc (int): Number of cross validations to perform
            k_splits (float): Number of k folds to make in the data. Optional,
                default is 4 folds.
            test_percent (float): Percentage of data set to set aside for 
                testing. Optional, default is 0.2.
            X_test (ndarray): Design matrix test data, shape (N,p). Optional, 
                default is using 0.2 percent of data as test data.
            y_test (ndarray): Design matrix test data. Optional, default is 
                default is using 0.2 percent of data as test data.
            shuffle (bool): if True, will shuffle the data before splitting
        """

        # Checks if we have provided test data or not
        if isinstance(X_test, type(None)) and \
                isinstance(y_test, type(None)):

            # Splits X data and design matrix data
            X_train, X_test, y_train, y_test = \
                sk_modsel.train_test_split(self.X_data, self.y_data,
                                           test_size=test_percent,
                                           shuffle=shuffle)

        else:
            # If X_test and y_test is provided, we simply use those as test
            # values.
            X_train = self.X_data
            y_train = self.y_data
            X_test = cp.deepcopy(X_test)
            y_test = cp.deepcopy(y_test)

        # Splits dataset into managable k fold tests
        mc_test_size = X_train.shape[0] // k_splits

        # All possible indices available
        mc_indices = list(range(X_train.shape[0]))

        # Stores the test values from each k trained data set in an array
        beta_coefs = []
        self.y_pred_list = np.empty((y_test.shape[0], N_mc))

        for i_mc in tqdm(range(N_mc), desc="Monte Carlo Cross Validation"):

            # Gets retrieves indexes for MC-CV. No replacement.
            mccv_test_indexes = np.random.choice(mc_indices, mc_test_size)
            mccv_train_indices = np.array(
                list(set(mc_indices) - set(mccv_test_indexes)))

            # Sets up new data set
            # k_x_train = x_mc_train[mccv_train_indices]
            k_X_train = X_train[mccv_train_indices]
            k_y_train = y_train[mccv_train_indices]

            # Trains method bu fitting data
            self.reg.fit(k_X_train, k_y_train)

            y_predict = self.reg.predict(X_test)

            # Adds prediction and beta coefs
            self.y_pred_list[:, i_mc] = y_predict.ravel()
            beta_coefs.append(self.reg.coef_)

        # Mean Square Error, mean((y - y_approx)**2)
        _mse = (y_test - self.y_pred_list)**2
        self.mse = np.mean(np.mean(_mse, axis=1, keepdims=True))

        # Bias, (y - mean(y_approx))^2
        _mean_pred = np.mean(self.y_pred_list, axis=1, keepdims=True)
        _bias = y_test - _mean_pred
        self.bias = np.mean(_bias**2)

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        # _r2 = metrics.r2(y_test, self.y_pred_list, axis=0)

        self.r2 = sk_metrics.r2_score(y_test, self.y_pred_list.mean(axis=1))
        # self.r2 = np.mean(_r2)

        # Variance, var(y_predictions)
        self.var = np.mean(np.var(self.y_pred_list, axis=1, keepdims=True))

        beta_coefs = np.asarray(beta_coefs)
        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=1)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=1)

        self.x_pred_test = X_test[:, 1]
        self.y_pred = np.mean(self.y_pred_list, axis=1)
        self.y_pred_var = np.var(self.y_pred_list, axis=1)


def kFoldCVWrapper(X, y, reg, k=4, test_percent=0.4,
                   shuffle=False, X_test=None, y_test=None):
    """k-fold Cross Validation using a manual method.

    Args:
        X_data (ndarray): design matrix on the shape (N, p)
        y_data (ndarray): y data on the shape (N, 1). Data to be 
            approximated.
        reg (Regression Instance): an initialized regression method
        k (int): optional, number of k folds. Default is 4.
        test_percent (float): optional, size of testing data. Default is 0.4.
        shuffle (bool): optional, if the data will be shuffled. Default is 
            False.
        X_test, (ndarray): design matrix for test values, shape (N, p).
        y_test, (ndarray): y test data on shape (N, 1).

    Return:
        dictionary with r2, mse, bias, var, coef, coef_var
    """

    kfcv_reg = kFoldCrossValidation(X, y, reg)
    kfcv_reg.cross_validate(k_splits=k, test_percent=test_percent,
                            shuffle=shuffle, X_test=X_test, y_test=y_test)

    return {
        "r2": kfcv_reg.r2, "mse": kfcv_reg.mse, "bias": kfcv_reg.bias,
        "var": kfcv_reg.var,
        "diff": kfcv_reg.mse - kfcv_reg.bias - kfcv_reg.var,
        "coef": kfcv_reg.beta_coefs, "coef_var": kfcv_reg.beta_coefs_var}
    # , "x_pred": kfcv_reg.x_pred_test,
    # "y_pred": kfcv_reg.y_pred, "y_pred_var": kfcv_reg.y_pred_var}


def SKLearnkFoldCV(X, y, reg, k=4, test_percent=0.4,
                   shuffle=False, X_test=None, y_test=None):
    """k-fold Cross Validation using SciKit Learn.

    Args:
        X_data (ndarray): design matrix on the shape (N, p)
        y_data (ndarray): y data on the shape (N, 1). Data to be 
            approximated.
        reg (Regression Instance): an initialized regression method
        k (int): number of k folds. Optional, default is 4.
        test_percent (float): size of testing data. Optional, default is 0.4.
        X_test, (ndarray): design matrix for test values, shape (N, p).
        y_test, (ndarray): y test data on shape (N, 1).

    Return:
        dictionary with r2, mse, bias, var, coef, coef_var
    """

    # kfcv_reg = kFoldCrossValidation(x, y, reg, design_matrix)
    # kfcv_reg.cross_validate(k_splits=k, test_percent=test_percent)

    if ((isinstance(X_test, type(None))) and
            (isinstance(y_test, type(None)))):

        # Splits X data and design matrix data
        X_train, X_test, y_train, y_test = \
            sk_modsel.train_test_split(X, y, test_size=test_percent,
                                       shuffle=False)

    else:
        # If X_test and y_test is provided, we simply use those as test values
        X_train = X
        y = y

    # X_train, X_test, y_train, y_test = sk_modsel.train_test_split(
    #     X, y, test_size=test_percent, shuffle=shuffle)

    # Preps lists to be filled
    y_pred_list = np.empty((y_test.shape[0], k))
    r2_list = np.empty(k)
    beta_coefs = []

    # Specifies the number of splits
    kfcv = sk_modsel.KFold(n_splits=k, shuffle=shuffle)

    for i, val in tqdm(enumerate(kfcv.split(X_train)),
                       desc="SK-learn k-fold CV"):

        train_index, test_index = val

        reg.fit(X_train[train_index], y_train[train_index])

        y_predict = reg.predict(X_test)

        r2_list[i] = sk_metrics.r2_score(y_test, y_predict)

        y_pred_list[:, i] = y_predict.ravel()
        beta_coefs.append(reg.coef_)

    # Mean Square Error, mean((y - y_approx)**2)
    _mse = (y_test - y_pred_list)**2
    mse = np.mean(np.mean(_mse, axis=1, keepdims=True))

    # Bias, (y - mean(y_approx))^2
    _mean_pred = np.mean(y_pred_list, axis=1, keepdims=True)
    _bias = y_test - _mean_pred
    bias = np.mean(_bias**2)

    # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
    r2 = r2_list.mean()

    # Variance, var(y_predictions)
    var = np.mean(np.var(y_pred_list, axis=1, keepdims=True))

    r2 = sk_metrics.r2_score(y_test, y_pred_list.mean(axis=1))

    return {"r2": r2, "mse": mse, "bias": bias, "var": var,
            "diff": mse - bias - var,
            "coef": np.asarray(beta_coefs).var(axis=1),
            "coef_var": np.asarray(beta_coefs).mean(axis=1)}


def kkfoldCVWrapper(X, y, reg, k=4, test_percent=0.4,
                    shuffle=False, X_test=None, y_test=None):
    """k-fold Cross Validation using a manual method.

    Args:
        X_data (ndarray): design matrix on the shape (N, p)
        y_data (ndarray): y data on the shape (N, 1). Data to be 
            approximated.
        reg (Regression Instance): an initialized regression method
        k (int): optional, number of k folds. Default is 4.
        test_percent (float): optional, size of testing data. Default is 0.4.
        shuffle (bool): optional, if the data will be shuffled. Default is 
            False.
        X_test, (ndarray): design matrix for test values, shape (N, p).
        y_test, (ndarray): y test data on shape (N, 1).

    Return:
        dictionary with r2, mse, bias, var, coef, coef_var
    """

    kkfcv_reg = kkFoldCrossValidation(X, y, reg)
    kkfcv_reg.cross_validate(k_splits=k, test_percent=test_percent,
                             shuffle=shuffle, X_test=X_test, y_test=y_test)

    return {
        "r2": kkfcv_reg.r2, "mse": kkfcv_reg.mse, "bias": kkfcv_reg.bias,
        "var": kkfcv_reg.var,
        "diff": kkfcv_reg.mse - kkfcv_reg.bias - kkfcv_reg.var,
        "coef": kkfcv_reg.beta_coefs, "coef_var": kkfcv_reg.beta_coefs_var}


def MCCVWrapper(X, y, reg, N_mc, k=4, test_percent=0.4, shuffle=False,
                X_test=None, y_test=None):
    """k-fold Cross Validation using a manual method.

    Args:
        X_data (ndarray): design matrix on the shape (N, p)
        y_data (ndarray): y data on the shape (N, 1). Data to be 
            approximated.
        reg (Regression Instance): an initialized regression method
        N_mc (int): number of MC samples to use.
        k (int): optional, number of k folds. Default is 4.
        test_percent (float): optional, size of testing data. Default is 0.4.
        shuffle (bool): optional, if the data will be shuffled. Default is 
            False.
        X_test, (ndarray): design matrix for test values, shape (N, p).
        y_test, (ndarray): y test data on shape (N, 1).

    Return:
        dictionary with r2, mse, bias, var, coef, coef_var
    """

    mccv_reg = MCCrossValidation(X, y, reg)
    mccv_reg.cross_validate(N_mc, k_splits=k, test_percent=test_percent,
                            X_test=X_test, y_test=y_test, shuffle=shuffle)

    return {
        "r2": mccv_reg.r2, "mse": mccv_reg.mse, "bias": mccv_reg.bias,
        "var": mccv_reg.var,
        "diff": mccv_reg.mse - mccv_reg.bias - mccv_reg.var,
        "coef": mccv_reg.beta_coefs, "coef_var": mccv_reg.beta_coefs_var}


def SKLearnMCCV(X, y, reg, N_bs, k=4, test_percent=0.4):
    raise NotImplementedError("SKLearnMCCV")


def __compare_kfold_cv():
    """Runs a comparison between implemented method of k-fold Cross Validation
    and SK-learn's implementation of SK-learn. Since they both are 
    deterministic, should the answer be exactly the same."""
    from regression import OLSRegression
    import sklearn.preprocessing as sk_preproc
    import sklearn.linear_model as sk_model
    import copy as cp

    deg = 2
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)

    k_splits = 4

    # N_bs = 10000

    # Initial values
    n = 100
    noise = 0.3
    np.random.seed(1234)
    test_percent = 0.35
    shuffle = False

    # Sets up random matrices
    x = np.random.rand(n, 1)
    # x = np.c_[np.linspace(0,1,n)]

    def func_excact(_x):
        return 2*_x*_x + np.exp(-2*_x)  # + noise * \
        #np.random.randn(_x.shape[0], _x.shape[1])

    y = func_excact(x)
    X = poly.fit_transform(x)

    kfcv_my = kFoldCVWrapper(
        cp.deepcopy(X), cp.deepcopy(y),
        sk_model.LinearRegression(fit_intercept=False), k=k_splits,
        test_percent=test_percent, shuffle=shuffle)

    print("Manual implementation:")
    print("r2:", kfcv_my["r2"], "mse:", kfcv_my["mse"],
          "var: {:.16f}".format(kfcv_my["var"]),
          "bias: {:.16f}".format(kfcv_my["bias"]),
          "diff: {:.16f}".format(
        abs(kfcv_my["mse"] - kfcv_my["var"] - kfcv_my["bias"])))

    kfcv_sk = SKLearnkFoldCV(
        cp.deepcopy(X), cp.deepcopy(y),
        sk_model.LinearRegression(fit_intercept=False), k=k_splits,
        test_percent=test_percent, shuffle=shuffle)

    print("SK-Learn:")
    print("r2:", kfcv_sk["r2"], "mse:", kfcv_sk["mse"],
          "var: {:.16f}".format(kfcv_sk["var"]),
          "bias: {:.16f}".format(kfcv_sk["bias"]),
          "diff: {:.16f}".format(
        abs(kfcv_sk["mse"] - kfcv_sk["var"] - kfcv_sk["bias"])))


def __compare_mc_cv():
    raise NotImplementedError("__compare_mc_cv")


def __time_cross_validations():
    """Timing the methods."""
    import time
    from regression import OLSRegression
    import sklearn.preprocessing as sk_preproc

    # Initial values
    n = 1000000
    N_bs = 4
    deg = 2
    k_splits = 8
    numprocs = 4
    test_percent = 0.35
    noise = 0.3
    np.random.seed(1234)
    # Sets up random matrices
    x = np.random.rand(n, 1)

    y = 2*x*x + np.exp(-2*x) + noise*np.random.randn(x.shape[0], x.shape[1])

    # Sets up design matrix
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(x)

    # k-fold Cross validation
    t0_kfcv_mp = time.time()
    kfcv = kFoldCrossValidation(X, y, OLSRegression(), numprocs=numprocs)
    kfcv.cross_validate(k_splits=k_splits,
                        test_percent=test_percent)
    t1_kfcv_mp = time.time()

    # k-fold Cross validation no parallelization
    t0_kfcv = time.time()
    kfcv = kFoldCrossValidation(X, y, OLSRegression(), numprocs=1)
    kfcv.cross_validate(k_splits=k_splits,
                        test_percent=test_percent)
    t1_kfcv = time.time()

    # k-k-fold Cross validation
    t0_kkfcv = time.time()
    kkfcv = kkFoldCrossValidation(X, y, OLSRegression())
    kkfcv.cross_validate(k_splits=k_splits,
                         test_percent=test_percent)
    t1_kkfcv = time.time()

    # mc Cross validation
    t0_mccv = time.time()
    mccv = MCCrossValidation(X, y, OLSRegression())
    mccv.cross_validate(N_bs, k_splits=k_splits,
                        test_percent=test_percent)
    t1_mccv = time.time()

    print("**Non-parallel**")
    print("Time taken for kf-cv:  {:.8f} seconds".format(t1_kfcv-t0_kfcv))
    print("Time taken for kkf-cv: {:.8f} seconds".format(t1_kkfcv-t0_kkfcv))
    print("Time taken for mc-cv:  {:.8f} seconds".format(t1_mccv-t0_mccv))
    print("**Parallelized**")
    print("Time taken for kf-cv:  {:.8f} seconds".format(t1_kfcv_mp-t0_kfcv_mp))


def __test_cross_validation_methods():
    # A small implementation of a test case
    from regression import OLSRegression
    import sklearn.preprocessing as sk_preproc
    import matplotlib.pyplot as plt

    # Initial values
    n = 1000
    N_bs = 200
    deg = 2
    k_splits = 4
    test_percent = 0.35
    noise = 0.3
    np.random.seed(1234)
    # Sets up random matrices
    x = np.random.rand(n, 1)

    y = 2*x*x + np.exp(-2*x) + noise*np.random.randn(x.shape[0], x.shape[1])

    # Sets up design matrix
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(x)

    # Performs regression
    reg = OLSRegression()
    reg.fit(X, y)
    y_predict = reg.predict(X)
    print("Regular linear regression")
    print("R2:    {:-20.16f}".format(reg.score(X, y)))
    print("MSE:   {:-20.16f}".format(metrics.mse(y, y_predict)))
    print("Bias^2:{:-20.16f}".format(metrics.bias(y, y_predict)))

    # Small plotter
    plt.plot(x, y, "o", label="data")
    plt.plot(x, y_predict, "o",
             label=r"Pred, $R^2={:.4f}$".format(reg.score(X, y)))

    print("k-fold Cross Validation")
    kfcv = kFoldCrossValidation(X, y, OLSRegression())
    kfcv.cross_validate(k_splits=k_splits,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(kfcv.r2))
    print("MSE:   {:-20.16f}".format(kfcv.mse))
    print("Bias^2:{:-20.16f}".format(kfcv.bias))
    print("Var(y):{:-20.16f}".format(kfcv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(kfcv.mse, kfcv.bias, kfcv.var,
                                     kfcv.bias + kfcv.var))
    print("Diff: {}".format(abs(kfcv.bias + kfcv.var - kfcv.mse)))

    plt.errorbar(kfcv.x_pred_test, kfcv.y_pred,
                 yerr=np.sqrt(kfcv.y_pred_var), fmt="o",
                 label=r"k-fold CV, $R^2={:.4f}$".format(kfcv.r2))

    print("kk Cross Validation")
    kkcv = kkFoldCrossValidation(X, y, OLSRegression())
    kkcv.cross_validate(k_splits=k_splits,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(kkcv.r2))
    print("MSE:   {:-20.16f}".format(kkcv.mse))
    print("Bias^2:{:-20.16f}".format(kkcv.bias))
    print("Var(y):{:-20.16f}".format(kkcv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(kkcv.mse, kkcv.bias, kkcv.var,
                                     kkcv.bias + kkcv.var))
    print("Diff: {}".format(abs(kkcv.bias + kkcv.var - kkcv.mse)))

    plt.errorbar(kkcv.x_pred_test, kkcv.y_pred,
                 yerr=np.sqrt(kkcv.y_pred_var), fmt="o",
                 label=r"kk-fold CV, $R^2={:.4f}$".format(kkcv.r2))

    print("Monte Carlo Cross Validation")
    mccv = MCCrossValidation(X, y, OLSRegression())
    mccv.cross_validate(N_bs, k_splits=k_splits,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(mccv.r2))
    print("MSE:   {:-20.16f}".format(mccv.mse))
    print("Bias^2:{:-20.16f}".format(mccv.bias))
    print("Var(y):{:-20.16f}".format(mccv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(mccv.mse, mccv.bias, mccv.var,
                                     mccv.bias + mccv.var))
    print("Diff: {}".format(abs(mccv.bias + mccv.var - mccv.mse)))

    print("\nCross Validation methods tested.")

    plt.errorbar(mccv.x_pred_test, mccv.y_pred,
                 yerr=np.sqrt(mccv.y_pred_var), fmt="o",
                 label=r"MC CV, $R^2={:.4f}$".format(mccv.r2))

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(r"$y=2x^2 + e^{-2x}$")
    y = 2*x*x + np.exp(-2*x) + noise*np.random.randn(x.shape[0], x.shape[1])
    plt.legend()
    # plt.show()


def __test_bias_variance_kfcv():
    """Checks bias-variance relation."""
    from regression import OLSRegression
    import sklearn.linear_model as sk_model
    import sklearn.preprocessing as sk_preproc
    import matplotlib.pyplot as plt

    # Initial values
    N_polynomials = 30
    deg_list = np.linspace(1, N_polynomials, N_polynomials, dtype=int)
    n = 500
    test_percent = 0.2
    noise = 0.1
    np.random.seed(2018)

    x = np.random.rand(n, 1)
    y = 2*x*x + np.exp(-2*x) + noise * \
        np.random.randn(x.shape[0], x.shape[1])

    x_train, x_test, y_train, y_test = \
        sk_modsel.train_test_split(x, y,
                                   test_size=test_percent,
                                   shuffle=False)

    mse_list = np.empty(N_polynomials)
    var_list = np.empty(N_polynomials)
    bias_list = np.empty(N_polynomials)
    r2_list = np.empty(N_polynomials)

    for i, deg in enumerate(deg_list):
        # Sets up design matrix
        poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
        X = poly.fit_transform(x_train)

        results = kFoldCVWrapper(X, y_train, sk_model.LinearRegression(
            fit_intercept=False),
            X_test=poly.fit_transform(x_test),
            y_test=y_test)

        mse_list[i] = results["mse"]
        var_list[i] = results["var"]
        bias_list[i] = results["bias"]
        r2_list[i] = results["r2"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(deg_list, mse_list, "-*", label=r"$\mathrm{MSE}$")
    ax.plot(deg_list, var_list, "-x", label=r"$\mathrm{Var}$")
    ax.plot(deg_list, bias_list, "-.", label=r"$\mathrm{Bias}$")
    ax.set_xlabel(r"Polynomial degree")
    ax.set_ylabel(r"MSE/Var/Bias")
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    # plt.show()
    plt.close(fig)


def __test_bias_variance_kkfcv():
    # """Checks bias-variance relation."""
    from regression import OLSRegression
    # import sklearn.linear_model as sk_model
    # import sklearn.preprocessing as sk_preproc
    # import matplotlib.pyplot as plt

    # Initial values
    N_polynomials = 30
    deg_list = np.linspace(1, N_polynomials, N_polynomials, dtype=int)
    n = 500
    test_percent = 0.2
    noise = 0.1
    np.random.seed(2018)

    x = np.random.rand(n, 1)
    y = 2*x*x + np.exp(-2*x) + noise * \
        np.random.randn(x.shape[0], x.shape[1])

    x_train, x_test, y_train, y_test = \
        sk_modsel.train_test_split(x, y,
                                   test_size=test_percent,
                                   shuffle=False)

    mse_list = np.empty(N_polynomials)
    var_list = np.empty(N_polynomials)
    bias_list = np.empty(N_polynomials)
    r2_list = np.empty(N_polynomials)

    for i, deg in enumerate(deg_list):
        # Sets up design matrix
        poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
        X = poly.fit_transform(x_train)

        results = kkfoldCVWrapper(X, y_train, sk_model.LinearRegression(
            fit_intercept=False),
            X_test=poly.fit_transform(x_test),
            y_test=y_test)

        mse_list[i] = results["mse"]
        var_list[i] = results["var"]
        bias_list[i] = results["bias"]
        r2_list[i] = results["r2"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(deg_list, mse_list, "-*", label=r"$\mathrm{MSE}$")
    ax.plot(deg_list, var_list, "-x", label=r"$\mathrm{Var}$")
    ax.plot(deg_list, bias_list, "-.", label=r"$\mathrm{Bias}$")
    ax.set_xlabel(r"Polynomial degree")
    ax.set_ylabel(r"MSE/Var/Bias")
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    # plt.show()
    plt.close(fig)


def __test_bias_variance_mccv():
    # """Checks bias-variance relation."""
    from regression import OLSRegression
    # import sklearn.linear_model as sk_model
    # import sklearn.preprocessing as sk_preproc
    # import matplotlib.pyplot as plt

    # Initial values
    N_polynomials = 30
    deg_list = np.linspace(1, N_polynomials, N_polynomials, dtype=int)
    n = 500
    N_bs = 200
    test_percent = 0.2
    noise = 0.1
    np.random.seed(2018)

    x = np.random.rand(n, 1)
    y = 2*x*x + np.exp(-2*x) + noise * \
        np.random.randn(x.shape[0], x.shape[1])

    x_train, x_test, y_train, y_test = \
        sk_modsel.train_test_split(x, y,
                                   test_size=test_percent,
                                   shuffle=False)

    mse_list = np.empty(N_polynomials)
    var_list = np.empty(N_polynomials)
    bias_list = np.empty(N_polynomials)
    r2_list = np.empty(N_polynomials)

    for i, deg in enumerate(deg_list):
        # Sets up design matrix
        poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
        X = poly.fit_transform(x_train)

        results = MCCVWrapper(X, y_train, sk_model.LinearRegression(
            fit_intercept=False), N_bs,
            X_test=poly.fit_transform(x_test), y_test=y_test)

        mse_list[i] = results["mse"]
        var_list[i] = results["var"]
        bias_list[i] = results["bias"]
        r2_list[i] = results["r2"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(deg_list, mse_list, "-*", label=r"$\mathrm{MSE}$")
    ax.plot(deg_list, var_list, "-x", label=r"$\mathrm{Var}$")
    ax.plot(deg_list, bias_list, "-.", label=r"$\mathrm{Bias}$")
    ax.set_xlabel(r"Polynomial degree")
    ax.set_ylabel(r"MSE/Var/Bias")
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    # plt.show()
    plt.close(fig)


if __name__ == '__main__':
    # __test_cross_validation_methods()
    # __test_bias_variance_kfcv()
    # __test_bias_variance_kkfcv()
    # __test_bias_variance_mccv()
    # __compare_kfold_cv()
    __time_cross_validations()
