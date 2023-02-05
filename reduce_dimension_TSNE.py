import numpy as np
from sklearn.manifold import TSNE


def reduce_dimension_TSNE(X, dim, scaling_factor=1):
    """
    Function wrapper for sklearn TSNE algorithm for reducing the dimension of the datapoints
    :param X:
    :param dim:
    :param scaling_factor:
    :return:
    """
    scaler = scaling_factor * np.identity(X.shape[1])
    X = np.array([np.matmul(scaler, x) for x in X])
    tsne_model = TSNE(perplexity=40, n_components=dim, init='pca', n_iter=2500, random_state=23)

    return tsne_model.fit_transform(X)
