import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)
    
    # shuffle the order
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    # keep the original order
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

# Calculate the loss.
def compute_loss(y, tx, w):
    e = y - np.dot(tx, w)
    loss_result_mse = 0.5 * np.mean(e**2)
    return loss_result_mse


# calculate the gradient for linear regression
def compute_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    N = np.shape(y)[0]
    gradient = (-1/N) * np.dot(tx.T, e)
    return gradient


# Compute a stochastic gradient from just few examples n and their corresponding y_n labels.
def compute_stoch_gradient(y, tx, w):
    grad = compute_gradient(y, tx, w)
    return grad


# apply the sigmoid function on t
def sigmoid(t):
    return 1/(1+np.exp(-t))


# compute the loss: negative log likelihood -> loss
def negative_log_likelihood(y, tx, w): 
    # L(w) = -sum_{i=1}^n{ y_i*log(sigmoid(x_i^T*w)) + (1-y_i)*log(1-sigmoid(x_i^T*w))}
    tx_w = sigmoid(np.dot(tx, w))
    first_part = np.dot(y.T, np.log(tx_w))
    second_part = np.dot((1-y.T), np.log(1-tx_w))
    return -(first_part+second_part)

# compute the gradient of loss -> logistic
def calculate_gradient_logistic(y, tx, w):
    # grad_L(w) = sum_{i=1}^n(sigmoid(x_i^T*w)-y_i)*x_i
    return tx.T.dot(sigmoid(tx.dot(w))-y)


# Do one step of gradient descent using logistic regression
def learning_by_gradient_descent(y, tx, w, gamma):
    loss = negative_log_likelihood(y, tx, w)
    gradient = calculate_gradient_logistic(y, tx, w)
    w = w - gamma * gradient
    return loss, w


# return the Hessian of the loss function
def calculate_hessian(y, tx, w):
    xw = sigmoid(np.dot(tx, w))
    S = (xw*(1-xw))
    S = np.diag(S.T[0])
    XTS = np.dot(tx.T, S)
    XTSX = np.dot(XTS, tx)
    return XTSX


# calculate the loss and gradient from the penalized_logistic_regression
def penalized_logistic_regression(y, tx, w, lambda_):
    penality = lambda_*np.linalg.norm(w)**2
    # diag = np.diag(np.repeat(2*lambda_, len(w)))
    loss = negative_log_likelihood(y, tx, w) + penality
    # print("lambda_: ", np.shape(lambda_))
    gradient = calculate_gradient_logistic(y, tx, w) + 2*lambda_*w
    return loss, gradient


def learning_by_penalized_logistic_gradient(y, tx, w, gamma, lambda_):
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return loss, w
