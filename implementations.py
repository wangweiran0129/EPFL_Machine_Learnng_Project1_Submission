# -*- coding: utf-8 -*-

# The following two packages are the only required ones used for model calculation
import numpy as np
from proj1_helpers import *
from preprocess import *

# IMPORTANT!!
# w computed from gradient_descent diverge!
# 1 Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''Gradient descent algorithm.'''
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        grad = compute_gradient(y, tx, w)
        # update w by gradient
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return losses, ws

# 2 Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    '''Stochastic gradient descent algorithm.'''
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size = batch_size):
            loss = compute_loss(y, tx, w)
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            ws.append(w)
            losses.append(loss)
    return losses, ws


# 3 Least squares regression using normal equations
def least_squares(y, tx):
    '''calculate the least squares solution.'''
    # w* = (X^T*X)^{-1} * X^T * y
    # x^T * x * w = x^T * y
    optimal_w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    # compute_loss(y, tx, w, flag = 0)
    mse = compute_loss(y, tx, optimal_w)
    return mse, optimal_w


# 4 ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    for ind, lambda_ in enumerate(lambda_):
        # left_part * w_ridge = right_part
        left_part = np.dot(tx.T, tx) + lambda_ * 2 * np.shape(tx)[0] * np.eye(np.shape(tx)[1])
        # print("lambda here is ", np.dot(lambda_ * 2 * np.shape(tx)[0], np.eye(np.shape(tx)[0])))
        # print("left part's shape: ", np.shape(left_part))
        right_part = np.dot(tx.T, y)
        #print("right part's shape: ", np.shape(right_part))
        w_ridge = np.linalg.solve(left_part, right_part)
        # w_ridge = np.dot(np.linalg.inv(np.dot(tx.T, tx) + lambda_), np.dot(tx.T, y))
        mse_ridge = compute_loss(y, tx, w_ridge)
    return mse_ridge, w_ridge


# 5 logistic_regression by gradient descent
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    w = initial_w
    for iter in range(max_iters):
        # get loss and update w.
        loss = negative_log_likelihood(y, tx, w)
        gradient = calculate_gradient_logistic(y, tx, w)
        w = w - gamma * gradient
    return loss, w


# 6 regularized logistic regression by gradient descent
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    losses = []
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_logistic_gradient(y, tx, w, gamma, lambda_)
        # w = w - gamma * gradient
    print("w's shape = ", np.shape(w))
    print("w = ", w)
    return loss, w
