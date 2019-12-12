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
dimension = 100
x_train, x_test = pca(x_train, x_test, dimension)

# alpha = 1
iteration = 500
best_alpha = 0.1
best_accuracy = 0
for alpha in np.arange(0.2, 2, 0.2):
    print('Current alpha: {}'.format(alpha))
    theta = np.zeros(shape=(x_train.shape[1], number_of_classes))
    theta_new = gradient_decent(theta, x_train, y_train_transform, alpha, iteration)
    # train_result = one_vs_all(theta_new, x_train)
    # print('Train accuracy: %.2f' % (accuracy(np.array(train_result), np.array(y_train)) * 100) + '%')
    test_result = one_vs_all(theta_new, x_test)
    current_accuracy = accuracy(np.array(test_result), np.array(y_test)) * 100
    print('Test accuracy: {:.2f}%\n'.format(current_accuracy))
    if best_accuracy < current_accuracy:
        best_accuracy = current_accuracy
        best_alpha = alpha
print("Best alpha: {}".format(best_alpha))
print("Best accuracy: {}%".format(best_accuracy))

