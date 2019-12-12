import numpy as np
from keras.datasets import mnist
from logistic_regression import accuracy, transform_y_for_one_vs_all, gradient_decent, one_vs_all, cost_function
from reduce_dimension import pca

number_of_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
y_train_transform = transform_y_for_one_vs_all(y_train, number_of_classes)
x_train = x_train / 255
x_test = x_test / 255
flag = False
way = int(input('Please choose way: Base logistic regression enter - 1 , With reduce dimension enter - 2: '))
print("x_normalize")
if way == 2:
    flag = True
# Dimension reduce
if flag:
    dimension = int(input('Please choose dimension: '))
    x_train, x_test = pca(x_train, x_test, dimension)

alpha = 1
iteration = 500
theta = np.zeros(shape=(x_train.shape[1], number_of_classes))
theta_new = gradient_decent(theta, x_train, y_train_transform, alpha, iteration)
cost = cost_function(y_train_transform, theta_new, x_train)
print("Final cost: {}".format(cost))
train_result = one_vs_all(theta_new, x_train)
test_result = one_vs_all(theta_new, x_test)
print('Train accuracy: %.2f' % (accuracy(np.array(train_result), np.array(y_train)) * 100) + '%')
print('Test accuracy: %.2f' % (accuracy(np.array(test_result), np.array(y_test)) * 100) + '%')
