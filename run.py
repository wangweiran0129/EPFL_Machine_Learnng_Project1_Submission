# -*- coding: utf-8 -*-

# The following two packages are the only required ones used for model calculation
import numpy as np
from numpy.lib.function_base import gradient
from proj1_helpers import *
from implementations import *
from cross_validation import *

# main is used for test here
def main():
    # choose methods
    print("Which methods do you want to implement with? (Please input the relevant integer) \n 1. least squares GD \n 2. least squares SGD \n 3. least squares \n 4. ridge regression \n 5. logistic regression \n 6. reg logistic regression")
    input_key = int(input())
    m = [1,2,3,4,5,6]
    try:
        temp = m.index(input_key)
    except ValueError:
        print("Invalid key! Please input an integer among [1, 6].")
    else:
        # training dataset
        # tx_train shape is (250000, 30)
        print("Start training...")
        y_train, tx_train_rough, ids_train = load_csv_data("train.csv")
        tx_train = standardize(tx_train_rough)[0]
        # print("tx_train's shape ", np.shape(tx_train))
        # print("tx_train ", tx_train)
        # testing dataset
        # tx_test shape is (568238, 30)
        print("Start testing...")
        y_test, tx_test, ids_test = load_csv_data("test.csv")

        w_initial = np.zeros(np.shape(tx_train)[1])
        max_iters = 5000
        gamma = 0.000001
        batch_size = 1
        lambdas = np.logspace(-5, 0, np.shape(tx_train)[1])
        if input_key == 1:
            # linear regression using gradient descent
            w_gradient = least_squares_GD(y_train, tx_train, w_initial, max_iters, gamma)[1]
            print(w_gradient)
            w = w_gradient[-1]
            #print("w_gradient = ", w_gradient)
            y_test = predict_labels(w, tx_test)
            create_csv_submission(ids_test, y_test, "sample-submission-least-squares-GD.csv")
            print("Test finished, please check the output file named 'sample-submission-least-squares-GD.csv' ")

        elif input_key == 2:
            # linear regression using stochastic gradient descent
            w_gradient = least_squares_SGD(y_train, tx_train, w_initial, batch_size, max_iters, gamma)[1]
            w = w_gradient[-1]
            y_test = predict_labels(w, tx_test)
            create_csv_submission(ids_test, y_test, "sample-submission-least-squares-SGD.csv")
            print("Test finished, please check the output file named 'sample-submission-least-squares-SGD.csv' ")

        elif input_key == 3:
            # least squares regression using normal equations
            w = least_squares(y_train, tx_train)[1]
            y_test = predict_labels(w, tx_test)
            create_csv_submission(ids_test, y_test, "sample-submission-least-squares.csv")
            print("Test finished, please check the output file named 'sample-submission-least-squares.csv' ")
            # visualization_test()

        elif input_key == 4:
            # ridge regression using normal equations
            mse_train, w = ridge_regression(y_train, tx_train, lambdas)
            y_test = predict_labels(w, tx_test)
            create_csv_submission(ids_test, y_test, "sample-submission-ridge-regression.csv")
            print("Test finished, please check the output file named 'sample-submission-ridge-regression.csv' ")

        elif input_key == 5:
            # logistic regression using gradient descent
            loss, w = logistic_regression(y_train, tx_train, w_initial, max_iters, gamma)
            y_test = predict_labels(w, tx_test)
            create_csv_submission(ids_test, y_test, "sample-submission-logistic-regression.csv")
            print("Test finished, please check the output file named 'sample-submission-logistic-regression.csv' ")

        else:
            # regularized logistic regression using gradient descent
            loss, w = reg_logistic_regression(y_train, tx_train, lambdas, w_initial, max_iters, gamma)
            y_test = predict_labels(w, tx_test)
            create_csv_submission(ids_test, y_test, "sample-submission-reg-logistic-regression.csv")
            print("Test finished, please check the output file named 'sample-submission-reg-logistic-regression.csv' ")


if __name__ == "__main__":
    main()
