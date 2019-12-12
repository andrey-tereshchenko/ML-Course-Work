import numpy as np


def accuracy(y_pred, y_true):
    return np.array(y_pred == y_true).mean()


def transform_y_for_one_vs_all(y, count_of_classes):
    Y = []
    for i in range(y.shape[0]):
        y_row = []
        for j in range(count_of_classes):
            if y[i] == j:
                y_row.append(1)
            else:
                y_row.append(0)
        Y.append(y_row)
    return np.array(Y)


def sigmoid(x):
    ones = np.ones(shape=(x.shape[0], 10))
    return ones / (ones + np.exp(x))


def h(theta, x):
    z = np.matmul(x, theta)
    return sigmoid(z)


def cost_function(y, theta, x):
    cost_vector = []
    for i in range(y.shape[1]):
        ones = np.ones(shape=(y.shape[0]))
        y_column = y[:, i]
        h_column = h(theta, x)[:, i]
        b = h_column
        a = ones - h_column
        cost = -y_column @ np.log(b) - (ones - y_column) @ np.log(a)
        cost_vector.append(cost)
    return cost_vector


def grad(theta, x, y):
    x_transpose = np.transpose(x)
    h1 = h(theta, x)
    grads = (1 / len(y)) * np.matmul(x_transpose, y - h1)
    return grads


def gradient_decent(theta, x, y, alpha, iteration):
    for i in range(iteration):
        # print("iter" + str(i))
        theta = theta - alpha * grad(theta, x, y)
    return theta


def one_vs_all(theta, x):
    h_matrix = h(theta, x)
    y_predict = np.argmax(h_matrix, axis=1)
    return y_predict
