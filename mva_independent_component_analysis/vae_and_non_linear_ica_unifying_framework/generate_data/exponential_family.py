import jax.numpy as jnp


def logdensity_function_ef(z, u, T, A, L):
    """
    Log-density function of an exponential distribution assuming
    univariate exponential distribution for each coordinate, i.e.,
    \pi(z\mid u) \coloneq \exp(\sum_{i=1}^d \sum_{j=1}^k T_{i,j}(z_i) L_{j,i}(u) - \sum_{i=1}^d A_j(u))
    for z scalar.
    :param z: scalar latent variable
    :param u: auxiliary variable
    :param T: sufficient statistics
    :param A: log-partition function
    :return: log-density
    """
    return jnp.einsum('ij,ji->', T(z), L(u)) - jnp.einsum('i->', A(u))


def logdensity_function_univ_normal(z_scalar, mu, sigma):
    """
    Log-density function of an univariate normal distribution
    :param z: scalar latent variable
    :param mu: mean
    :param sigma: standard deviation
    :return: log-density
    """
    return logdensity_function_multivariate_normal(z_scalar, mu.reshape(1, ), sigma.reshape((1, 1)))


def logdensity_function_multivariate_normal(z, mu, sigma):
    """
    Log-density function of a multivariate normal distribution
    with diagonal covariance matrix, constructed using the univariate normal distribution
    :param z: vector
    :param mu: mean vector
    :param sigma: diagonal variance matrix
    :return: log-density
    """
    sigma = jnp.diag(sigma)
    assert jnp.all(sigma > 0)

    def Ts(z):
        return jnp.vstack([z.T, (z ** 2).T]).T

    def As(u):
        return 0.5 * mu ** 2 / sigma ** 2 + jnp.log(jnp.sqrt(2 * jnp.pi) * sigma)

    def Ls(u):
        return jnp.vstack([(mu / sigma ** 2).T, (-1 / (2 * sigma ** 2)).T])

    return logdensity_function_ef(z, 0.0, Ts, As, Ls), Ts, As, Ls


def logdensity_laplace_distribution_univ(x, mu, b):
    """
    Log-density function of a univariate Laplace distribution
    :param mu: mean
    :param b: scale
    :return: log-density
    """
    return 1 / (2 * b) * jnp.exp(- jnp.abs(x - mu) / b)
