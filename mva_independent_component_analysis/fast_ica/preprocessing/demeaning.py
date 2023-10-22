import jax.numpy as jnp


def demeaning(X):
    """
    Centering the data.
    :param X: data matrix of shape (n_features, n_samples)
    :return: centered data matrix
    """
    return X - jnp.kron(jnp.ones(X.shape[1], ), jnp.mean(X, axis=1, keepdims=True))
