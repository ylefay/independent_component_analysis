import jax.numpy as jnp


def whitening(X):
    """
    Whitening the data using eigenvalue decomposition.
    :param X: centred data matrix of shape (n_features, n_samples)
    :return: whitened data matrix, i.e., with the identity matrix as covariance matrix.
    """
    cov = jnp.cov(X)
    eigenvalues, eigenvectors = jnp.linalg.eig(cov)
    eigenvalues = jnp.real(eigenvalues)
    eigenvectors = jnp.real(eigenvectors)
    return jnp.diag(1 / jnp.sqrt(eigenvalues)) @ eigenvectors.T @ X
