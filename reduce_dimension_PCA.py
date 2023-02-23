import numpy as np
from sklearn import decomposition


def reduce_dimension_PCA(X, dim, scaling_factor=1):
    """

    :param X:
    :param dim:
    :param scaling_factor:
    :return:
    """
    if scaling_factor != 1:
        scaler = scaling_factor * np.identity(X.shape[1])
        X = np.array([np.matmul(scaler, x) for x in X])
    pca = decomposition.PCA()
    pca.n_components = dim
    pca_X = pca.fit_transform(X)

    return pca_X
