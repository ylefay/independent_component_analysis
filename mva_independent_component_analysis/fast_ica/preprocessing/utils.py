import jax.numpy as jnp


def centering(X):
    """
    Centering the data.
    :param X: data matrix of shape (n_features, n_samples)
    :return: centered data matrix and the mean
    """
    mean = jnp.mean(X, axis=1, keepdims=True).reshape(X.shape[0], 1)
    return X - mean, mean


def whitening(X):
    """
    Whitening the data using eigenvalue decomposition.
    :param X: centred data matrix of shape (n_features, n_samples)
    :return: whitened data matrix, i.e., with the identity matrix as covariance matrix, and the corresponding linear transformation
    """
    cov = jnp.cov(X)
    if cov.shape == ():
        cov = jnp.array([[cov]])
    eigenvalues, eigenvectors = jnp.linalg.eig(cov)
    eigenvalues = jnp.real(eigenvalues)
    eigenvectors = jnp.real(eigenvectors)
    inverse_eigenvalues = jnp.diag(1 / jnp.sqrt(eigenvalues))
    return inverse_eigenvalues @ eigenvectors.T @ X, inverse_eigenvalues @ eigenvectors.T


def centering_and_whitening(X):
    """
    Centering and whitening the data using the two previous functions
    :param X: data matrix of shape (n_features, n_samples)
    :return: centered and whitened data matrix, X = m + sigma @ X
    """
    X, mean = centering(X)
    X, sigma = whitening(X)
    return X, mean, sigma
