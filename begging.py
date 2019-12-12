import numpy as np
from keras.datasets import mnist
from logistic_regression import accuracy, transform_y_for_one_vs_all, gradient_decent, one_vs_all, cost_function
from reduce_dimension import pca


def transform_dataset_to_begging(x, y, number_datasets, size):
    datasets = []
    datasets_y = []
    for i in range(number_datasets):
        idx = np.random.randint(x.shape[0], size=size)
        new_dataset = x[idx, :]
        dataset_y = y[idx, :]
        datasets.append(new_dataset)
        datasets_y.append(dataset_y)
    return datasets, datasets_y


def begging_train(datasets, theta, y, alpha, iterations):
    theta_list = []
    for i in range(len(datasets)):
        print('Gradient decent operating dataset: {}'.format(i + 1))
        dataset = datasets[i]
        new_theta = gradient_decent(theta, dataset, y[i], alpha, iterations)
        theta_list.append(new_theta)
    return theta_list


def begging_result(thetha_list, x):
    result_list = []
    for i in range(len(thetha_list)):
        current_theta = thetha_list[i]
        result = one_vs_all(current_theta, x)
        result_list.append(result)
    result_list = np.array(result_list).T
    result = [np.bincount(row).argmax() for row in result_list]
    return result


number_of_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
y_train_transform = transform_y_for_one_vs_all(y_train, number_of_classes)

x_train = x_train / 255
x_test = x_test / 255
print("x_normalize")
flag = False
way = int(input('Please choose way: Simple Begging enter - 1 , With reduce dimension enter - 2: '))
if way == 2:
    flag = True
# Dimension reduce
if flag:
    dimension = int(input('Please choose dimension: '))
    x_train, x_test = pca(x_train, x_test, dimension)

dataset_amount = 5
dataset_size = 30000
begging_datasets, datasets_y = transform_dataset_to_begging(x_train, y_train_transform, dataset_amount, dataset_size)

alpha = 1
iteration = 500
theta = np.zeros(shape=(begging_datasets[0].shape[1], number_of_classes))
theta_list = begging_train(begging_datasets, theta, datasets_y, alpha, iteration)
test_result = begging_result(theta_list, x_test)
print('Test accuracy: %.2f' % (accuracy(np.array(test_result), np.array(y_test)) * 100) + '%')
