#!/usr/bin/env python3
import numpy as np
import copy as cp
from tqdm import tqdm

try:
    import lib.metrics as metrics
except ModuleNotFoundError:
    import metrics

import sklearn.model_selection as sk_modsel
import sklearn.utils as sk_utils
import sklearn.metrics as sk_metrics

import multiprocessing


def _para_bs(X_train, y_train, X_test, y_test, reg):
    # Bootstraps test data
    # x_boot, y_boot = boot(x_train, y_train)

    X_boot, y_boot = boot(X_train, y_train)
    # X_boot, y_boot = sk_utils.resample(X_train, y_train)
    # Sets up design matrix
    # X_boot = self._design_matrix(x_boot)

    # Fits the bootstrapped values
    reg.fit(X_boot, y_boot)

    # Tries to predict the y_test values the bootstrapped model
    y_predict = reg.predict(X_test)

    return [
        sk_metrics.r2_score(y_test, y_predict),  # Calculates r2
        y_predict.ravel(),
        reg.coef_,  # Stores the prediction and beta coefs.
    ]


def boot(*data):
    """Strip-down version of the bootstrap method.

    Args:
        *data (ndarray): list of data arrays to resample.

    Return:
        *bs_data (ndarray): list of bootstrapped data arrays."""

    N_data = len(data)
    N = data[0].shape[0]
    # assert np.all(np.array([len(d) for d in data]) == N), \
    #     "unequal lengths of data passed."

    index_lists = np.random.randint(N, size=N)

    return [d[index_lists] for d in data]


class BootstrapRegression:
    """Bootstrap class intended for use together with regression."""
    _reg = None

    def __init__(self, X_data, y_data, reg):
        """
        Initialises an bootstrap regression object.

        Args:
            X_data (ndarray): Design matrix, on shape (N, p)
            y_data (ndarray): y data, observables, shape (N, 1)
            reg: regression method object. Must have method fit, predict 
                and coef_.
        """

        assert X_data.shape[0] == len(y_data), ("x and y data not of equal"
                                                " lengths")

        assert hasattr(reg, "fit"), ("regression method must have "
                                     "attribute fit()")
        assert hasattr(reg, "predict"), ("regression method must have "
                                         "attribute predict()")

        self.X_data = cp.deepcopy(X_data)
        self.y_data = cp.deepcopy(y_data)
        self._reg = reg

    @property
    def reg(self):
        return self._reg

    @reg.setter
    def reg(self, reg):
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

    @metrics.timing_function
    def bootstrap(self, N_bs, test_percent=0.25, X_test=None, y_test=None,
                  shuffle=False):
        """
        Performs a bootstrap for a given regression type, design matrix 
        function and excact function.

        Args:
            N_bs (int): number of bootstraps to perform
            test_percent (float): what percentage of data to reserve for 
                testing. optional, default is 0.25.
            X_test (ndarray): design matrix for test values, shape (N, p), 
                optional. Will use instead of splitting dataset by test 
                percent.
            y_test (ndarray): y test data on shape (N, 1), optional. Will 
                use instead of splitting dataset by test percent.
            shuffle (bool): if we are to shuffle the data when no test data 
                is provided. Default is False.
        """

        assert not isinstance(self._reg, type(None))

        assert test_percent < 1.0, "test_percent must be less than one."

        N = len(self.X_data)

        X = self.X_data
        y = self.y_data

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

        self.x_pred_test = X_test[:, 1]

        # Sets up emtpy lists for gathering the relevant scores in
        r2_list = np.empty(N_bs)
        beta_coefs = []

        y_pred_list = np.empty((X_test.shape[0], N_bs))

        # # Sets up jobs for parallel processing
        # input_values = list(zip([X_train for i in range(N_bs)],
        #                    [y_train for i in range(N_bs)],
        #                    [X_test for i in range(N_bs)],
        #                    [y_test for i in range(N_bs)],
        #                    [cp.deepcopy(self.reg) for i in range(N_bs)]))
        # print (input_values[0])
        # # Initializes multiprocessing
        # pool = multiprocessing.Pool(processes=4)

        # # Runs parallel processes. Can this be done more efficiently?
        # results = pool.map(_para_bs, input_values)
        # print (results)
        # # [sk_metrics.r2_score(y_test, y_predict),
        # #     y_predict.ravel(),
        # #     reg.coef_]

        # # Garbage collection for multiprocessing instance
        # pool.close()

        # Bootstraps
        for i_bs in tqdm(range(N_bs), desc="Bootstrapping"):
            # Bootstraps test data
            # x_boot, y_boot = boot(x_train, y_train)

            X_boot, y_boot = boot(X_train, y_train)
            # X_boot, y_boot = sk_utils.resample(X_train, y_train)
            # Sets up design matrix
            # X_boot = self._design_matrix(x_boot)

            # Fits the bootstrapped values
            self.reg.fit(X_boot, y_boot)

            # Tries to predict the y_test values the bootstrapped model
            y_predict = self.reg.predict(X_test)

            # Calculates r2
            # r2_list[i_bs] = metrics.r2(y_test, y_predict)
            r2_list[i_bs] = sk_metrics.r2_score(y_test, y_predict)

            # mse_list[i_bs] = metrics.mse(y_predict, y_test)
            # bias_list[i_bs] = metrics.bias(y_predict, y_test)
            # var_list[i_bs] = np.var(y_predict)

            # Stores the prediction and beta coefs.
            y_pred_list[:, i_bs] = y_predict.ravel()
            beta_coefs.append(self.reg.coef_)

        # pred_list_bs = np.mean(y_pred_list, axis=0)

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        self.r2 = np.mean(r2_list)

        # Mean Square Error, mean((y - y_approx)**2)
        # _mse = np.mean((y_test.ravel() - y_pred_list)**2,
        #                axis=0, keepdims=True)

        # _mse = np.mean((y_test - y_pred_list)**2,
        #                axis=1, keepdims=True)
        # self.mse = np.mean(_mse)

        # # Bias, (y - mean(y_approx))^2
        # _y_pred_mean = np.mean(y_pred_list, axis=1, keepdims=True)
        # self.bias = np.mean((y_test - _y_pred_mean)**2)

        # # Variance, var(y_approx)
        # self.var = np.mean(np.var(y_pred_list,
        #                           axis=1, keepdims=True))

        _y_pred_mean = np.mean(y_pred_list, axis=1, keepdims=True)
        self.bias = np.mean((y_test - _y_pred_mean)**2)

        # Variance, var(y_approx)
        self.var = np.mean(np.var(y_pred_list, axis=1, keepdims=True))

        self.mse = np.mean(
            np.mean((y_test - y_pred_list)**2, axis=1, keepdims=True))
        # self.bias = np.mean(
        #     y_test - np.mean(y_pred_list, axis=1, keepdims=True))**2

        beta_coefs = np.asarray(beta_coefs)

        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=0)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=0)

        self.y_pred = y_pred_list.mean(axis=1)
        self.y_pred_var = y_pred_list.var(axis=1)


def BootstrapWrapper(X_train, y_train, reg, N_bs, test_percent=0.4,
                     X_test=None, y_test=None, shuffle=False):
    """
    Wrapper for manual bootstrap method.
    """

    # Checks if we have provided test data or not
    if ((isinstance(X_test, type(None))) and
            (isinstance(y_test, type(None)))):

        # Splits X data and design matrix data
        X_train, X_test, y_train, y_test = \
            sk_modsel.train_test_split(X_train, y_train,
                                       test_size=test_percent,
                                       shuffle=shuffle)

    bs_reg = BootstrapRegression(X_train, y_train, reg)
    bs_reg.bootstrap(N_bs, test_percent=test_percent, X_test=X_test,
                     y_test=y_test)

    return {
        "r2": bs_reg.r2, "mse": bs_reg.mse, "bias": bs_reg.bias,
        "var": bs_reg.var, "diff": bs_reg.mse - bs_reg.bias - bs_reg.var,
        "coef": bs_reg.beta_coefs,
        "coef_var": bs_reg.beta_coefs_var, "x_pred": bs_reg.x_pred_test,
        "y_pred": bs_reg.y_pred, "y_pred_var": bs_reg.y_pred_var}


def SKLearnBootstrap(X_train, y_train, reg, N_bs, test_percent=0.4,
                     X_test=None, y_test=None, shuffle=False):
    """
    A wrapper for the Scikit-Learn Bootstrap method.
    """

    # Checks if we have provided test data or not
    if ((isinstance(X_test, type(None))) and
            (isinstance(y_test, type(None)))):

        # Splits X data and design matrix data
        X_train, X_test, y_train, y_test = \
            sk_modsel.train_test_split(X_train, y_train,
                                       test_size=test_percent,
                                       shuffle=shuffle)

    # Storage containers for results
    y_pred_array = np.empty((y_test.shape[0], N_bs))
    r2_array = np.empty(N_bs)
    mse_array = np.empty(N_bs)

    beta_coefs = []

    # for i_bs, val_ in enumerate(bs):
    for i_bs in tqdm(range(N_bs), desc="SKLearnBootstrap"):
        X_boot, y_boot = sk_utils.resample(X_train, y_train)
        # X_boot, y_boot = X_train[train_index], y_train[train_index]

        reg.fit(X_boot, y_boot)
        y_predict = reg.predict(X_test)
        y_pred_array[:, i_bs] = y_predict.ravel()

        r2_array[i_bs] = sk_metrics.r2_score(y_test, y_predict)
        # r2_array[i_bs] = metrics.r2(y_test, y_predict)
        # mse_array[i_bs] = sk_metrics.mean_squared_error(y_test, y_predict)

        beta_coefs.append(reg.coef_)

    # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
    r2 = np.mean(r2_array)

    # # Mean Square Error, mean((y - y_approx)**2)
    # _mse = np.mean((y_test.ravel() - y_pred_list)**2,
    #                axis=0, keepdims=True)
    # mse = np.mean(mse_array)
    _mse = np.mean((y_test - y_pred_array)**2,
                   axis=1, keepdims=True)
    mse = np.mean(_mse)

    # Bias, (y - mean(y_approx))^2
    _y_pred_mean = np.mean(y_pred_array, axis=1, keepdims=True)
    bias = np.mean((y_test - _y_pred_mean)**2)

    # Variance, var(y_approx)
    var = np.mean(np.var(y_pred_array, axis=1, keepdims=True))

    beta_coefs = np.asarray(beta_coefs)

    coef_var = np.asarray(beta_coefs).var(axis=0)
    coef_ = np.asarray(beta_coefs).mean(axis=0)

    X_pred_test = X_test
    y_pred = y_pred_array.mean(axis=1)
    y_pred_var = y_pred_array.var(axis=1)

    return {
        "r2": r2, "mse": mse, "bias": bias,
        "var": var, "diff": mse - bias - var,
        "coef": coef_, "coef_var": coef_var, "x_pred": X_test[:, 1],
        "y_pred": y_pred, "y_pred_var": y_pred_var}


def __test_bootstrap_fit():
    """A small implementation of a test case."""
    from regression import OLSRegression
    import sklearn.preprocessing as sk_preproc

    # Initial values
    deg = 2
    N_bs = 1000
    n = 100
    test_percent = 0.35
    noise = 0.3
    np.random.seed(1234)

    # Sets up random matrices
    x = np.random.rand(n, 1)

    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)

    y = 2*x*x + np.exp(-2*x) + noise * \
        np.random.randn(x.shape[0], x.shape[1])

    # Sets up design matrix
    X = poly.fit_transform(x)

    # Performs regression
    reg = OLSRegression()
    reg.fit(X, y)
    y_predict = reg.predict(X).ravel()
    print("Regular linear regression")
    print("r2:  {:-20.16f}".format(reg.score(X, y)))
    print("mse: {:-20.16f}".format(metrics.mse(y, reg.predict(X))))
    print("Beta:      ", reg.coef_.ravel())
    print("var(Beta): ", reg.coef_var.ravel())
    print("")

    # Performs a bootstrap
    print("Bootstrapping")
    bs_reg = BootstrapRegression(X, y, OLSRegression())
    bs_reg.bootstrap(N_bs, test_percent=test_percent)

    print("r2:    {:-20.16f}".format(bs_reg.r2))
    print("mse:   {:-20.16f}".format(bs_reg.mse))
    print("Bias^2:{:-20.16f}".format(bs_reg.bias))
    print("Var(y):{:-20.16f}".format(bs_reg.var))
    print("Beta:      ", bs_reg.coef_.ravel())
    print("var(Beta): ", bs_reg.coef_var.ravel())
    print("mse = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(bs_reg.mse, bs_reg.bias, bs_reg.var,
                                     bs_reg.bias + bs_reg.var))
    print("Diff: {}".format(abs(bs_reg.bias + bs_reg.var - bs_reg.mse)))

    import matplotlib.pyplot as plt
    plt.plot(x.ravel(), y, "o", label="Data")
    plt.plot(x.ravel(), y_predict, "o",
             label=r"Pred, R^2={:.4f}".format(reg.score(X, y)))
    plt.errorbar(bs_reg.x_pred_test, bs_reg.y_pred,
                 yerr=np.sqrt(bs_reg.y_pred_var), fmt="o",
                 label=r"Bootstrap Prediction, $R^2={:.4f}$".format(bs_reg.r2))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(r"$2x^2 + \sigma^2$")
    plt.legend()
    plt.show()


def __test_bias_variance_bootstrap():
    """Checks bias-variance relation."""
    from regression import OLSRegression
    import sklearn.preprocessing as sk_preproc
    import matplotlib.pyplot as plt
    import copy as cp

    import sklearn.linear_model as sk_model

    np.random.seed(2018)

    # Initial values
    n = 500
    N_bs = 100
    N_polynomials = 30
    deg_list = np.linspace(1, N_polynomials, N_polynomials, dtype=int)
    noise = 0.1
    test_percent = 0.2

    # x = np.random.rand(n, 1)
    # y = 2*x*x + np.exp(-2*x) + noise * \
    #     np.random.randn(x.shape[0], x.shape[1])

    # Piazza post values
    x = np.linspace(-1, 3, n).reshape(-1, 1)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + \
        np.random.normal(0, noise, x.shape)

    x_train, x_test, y_train, y_test = \
        sk_modsel.train_test_split(x, y,
                                   test_size=test_percent,
                                   shuffle=True)

    mse_list = np.empty(N_polynomials)
    var_list = np.empty(N_polynomials)
    bias_list = np.empty(N_polynomials)
    r2_list = np.empty(N_polynomials)

    for i, deg in enumerate(deg_list):

        print("Degree:", deg)

        # Sets up design matrix
        poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
        X = poly.fit_transform(cp.deepcopy(x_train))

        # SKLearnBootstrap.
        # BootstrapWrapper
        reg = OLSRegression()
        # reg = sk_model.LinearRegression(fit_intercept=False)
        results = SKLearnBootstrap(poly.fit_transform(cp.deepcopy(x_train)),
                                   cp.deepcopy(y_train), reg,
                                   N_bs,
                                   X_test=poly.fit_transform(
                                       cp.deepcopy(x_test)),
                                   y_test=cp.deepcopy(y_test))

        mse_list[i] = results["mse"]
        var_list[i] = results["var"]
        bias_list[i] = results["bias"]
        r2_list[i] = results["r2"]

        # print("My method:")
        # print('Error:', results["mse"])
        # print('Bias^2:', results["bias"])
        # print('Var:', results["var"])
        # print('{} >= {} + {} = {}'.format(
        #     results["mse"], results["bias"], results["var"],
        #     results["bias"]+results["var"]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(deg_list, mse_list, "-*", label=r"$\mathrm{MSE}$")
    ax.plot(deg_list, var_list, "-x", label=r"$\mathrm{Var}$")
    ax.plot(deg_list, bias_list, "-.", label=r"$\mathrm{Bias}$")
    ax.set_xlabel(r"Polynomial degree")
    ax.set_ylabel(r"MSE/Var/Bias")
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    plt.show()
    plt.close(fig)


def __test_basic_bootstrap():
    import matplotlib.pyplot as plt
    # Data to load and analyse
    data = np.random.normal(0, 2, 100)

    bs_data = np.empty((500, 100))
    # Histogram bins
    N_bins = 20

    # Bootstrapping
    N_bootstraps = int(500)
    for iboot in range(N_bootstraps):
        bs_data[iboot] = np.asarray(boot(data))

    print("Non-BS: {0:.16f} +/- {1:.16f}".format(data.mean(), data.std()))
    bs_data = bs_data.mean(axis=0)
    print("BS:     {0:.16f} +/- {1:.16f}".format(bs_data.mean(),
                                                 bs_data.std()))

    plt.hist(data, label=r"Data, ${0:.3f}\pm{1:.3f}$".format(
        data.mean(), data.std()))
    plt.hist(bs_data, label=r"Bootstrap, ${0:.3f}\pm{1:.3f}$".format(
        bs_data.mean(), bs_data.std()))
    plt.legend()
    plt.show()


def __test_compare_bootstrap_manual_sklearn():
    # Compare SK learn bootstrap and manual bootstrap
    from regression import OLSRegression
    import sklearn.linear_model as sk_model
    import sklearn.preprocessing as sk_preproc
    import copy as cp

    deg = 2
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)

    N_bs = 1000

    # Initial values
    n = 200
    noise = 0.1
    np.random.seed(1234)
    test_percent = 0.4

    # Sets up random matrices
    x = np.random.rand(n, 1)

    y = 2*x**2 + np.exp(-2*x) + noise *\
        np.random.randn(x.shape[0], x.shape[1])

    X = poly.fit_transform(x)

    bs_my = BootstrapWrapper(
        cp.deepcopy(X), cp.deepcopy(y), sk_model.LinearRegression(), N_bs,
        test_percent=test_percent)
    bs_sk = SKLearnBootstrap(
        cp.deepcopy(X), cp.deepcopy(y), sk_model.LinearRegression(), N_bs,
        test_percent=test_percent)

    print("Manual:")
    print("r2:", bs_my["r2"], "mse:", bs_my["mse"], "var:", bs_my["var"],
          "bias:", bs_my["bias"], bs_my["mse"] - bs_my["var"] - bs_my["bias"])
    print("Scikit-Learn:")
    print("r2:", bs_sk["r2"], "mse:", bs_sk["mse"], "var:", bs_sk["var"],
          "bias:", bs_sk["bias"], bs_sk["mse"] - bs_sk["var"] - bs_sk["bias"])

    import matplotlib.pyplot as plt
    plt.plot(x, y, ".", label="original")
    plt.plot(bs_my["x_pred"], bs_my["y_pred"], "x", label="sklearn")
    plt.plot(bs_sk["x_pred"], bs_sk["y_pred"], ">", label="manual")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # __test_bootstrap_fit()
    # __test_bias_variance_bootstrap()
    # __test_basic_bootstrap()
    __test_compare_bootstrap_manual_sklearn()
