import jax.numpy as jnp


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
