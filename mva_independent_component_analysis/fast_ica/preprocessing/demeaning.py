import jax.numpy as jnp


def demeaning(X):
    """
    Centering the data.
    :param X: data matrix of shape (n_features, n_samples)
    :return: centered data matrix and the mean
    """
    mean = jnp.mean(X, axis=1, keepdims=True).reshape(X.shape[0], 1)
    return X - mean, mean
