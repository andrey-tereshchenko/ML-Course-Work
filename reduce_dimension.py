import numpy as np


def pca(x, x_test, dimension):
    # x_c = x - x.mean(axis=0)
    # x_test_c = x_test - x_test.mean(axis=0)
    x_c = x
    x_test_c = x_test
    C = (x_c.T @ x_c) / (x_c.shape[0] - 1)
    L, W = np.linalg.eig(C)
    importance_of_components = L / L.sum()
    print('Remain information from dataset:{:.2f}%'.format(importance_of_components[:dimension].sum() * 100))
    important_indexes = np.argsort(importance_of_components)[-dimension:][::-1]
    x_projected = x_c @ W[:, important_indexes]
    x_projected_test = x_test_c @ W[:, important_indexes]
    return x_projected, x_projected_test
