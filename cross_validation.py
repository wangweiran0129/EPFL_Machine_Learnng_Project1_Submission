import numpy as np
from preprocess import *
import matplotlib as plt

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    plt.show()


def ridge_regression_plot(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    # catch the k'th subgroup in test group and catch others in train group
    te_ind = k_indices[k]
    tr_ind = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_ind = tr_ind.reshape(-1)
    y_test = y[te_ind]
    y_train = y[tr_ind]
    x_test = x[te_ind]
    x_train = x[tr_ind]
    # use polynomial degree to build data
    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)
    # derive w by ridge regression
    w = ridge_regression_plot(y_train, tx_train, lambda_)
    # calculate the loss for train and test data
    loss_train = np.sqrt(2 * compute_loss(y_train, tx_train, w))
    loss_test = np.sqrt(2 * compute_loss(y_test, tx_test, w))
    return loss_train, loss_test


def cross_validation_demo(y, x):
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_train = []
    rmse_test = []
    for lambda_ in lambdas:
        tmp_rmse_train = []
        tmp_rmse_test = []
        for k in range(k_fold):
            loss_train, loss_test = cross_validation(y, x, k_indices, k, lambda_, degree)
            tmp_rmse_train.append(loss_train)
            tmp_rmse_test.append(loss_test)
        rmse_train.append(np.mean(tmp_rmse_train))
        rmse_test.append(np.mean(tmp_rmse_test))
    cross_validation_visualization(lambdas, rmse_train, rmse_test)
