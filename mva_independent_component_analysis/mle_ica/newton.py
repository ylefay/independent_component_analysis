import jax
import jax.numpy as jnp
from jax import lax


def subgaussian(x):
    """
    Subgaussian log probability of a single source x.
    Assumption:
            p_i propto Exp[-x^2/2 - Log[Cosh[x]]]
    i.e.,  g_i = x + tanh(x)
    :param x: an array of shape (n_samples, ), float.

    """
    return jnp.tanh(x) + x


def supergaussian(x):
    """
    Supergaussian log probability of a single source x.
    Assumption:
        p_i propto Exp[-x^2/2 + Log[Cosh[x]]]
    i.e.,  g_i = x - tanh(x)
    :param x: an array of shape (n_samples, ), float.

    """
    return x - jnp.tanh(x)


def switching_criterion_kurtosis(y):
    return jnp.mean(y ** 4) - 3


def random_semiorthogonal_matrix(op_key, shape):
    """
    Generate a random semiorthogonal matrix of shape (m, n),
    :param op_key: jax.random.PRNGKey
    :param m: int, number of rows
    :param n: int, number of columns
    :return: array of shape (m, n)
    """
    q, _ = jnp.linalg.qr(jax.random.normal(op_key, shape=shape), mode='reduced')
    return q


def newton_ica(op_key, X, n_components=None, tol=1e-8, max_iter=10 ** 5,
               learning_rate=0.001, prior=None):
    n_features, n_samples = X.shape
    if n_components is None:
        n_components = n_features

    def compute_update_matrix(X, W):
        S = W @ X
        id = jnp.eye(n_components, n_components)
        if prior is None:
            sign_matrix = jnp.diag(jax.vmap(switching_criterion_kurtosis)(S))
            dW = (id - 1 / n_samples * (sign_matrix @ jax.vmap(jnp.tanh)(S) - S) @ S.T) @ W
        if prior == 'super':
            dW = (id - 1 / n_samples * (jax.vmap(jnp.tanh)(S) - S) @ S.T) @ W
        if prior == 'sub':
            dW = (id + 1 / n_samples * (jax.vmap(jnp.tanh)(S) - S) @ S.T) @ W
        return dW

    def cond(args):
        step, _, update_matrix = args
        return (step < max_iter) & (jnp.linalg.norm(update_matrix, 'fro') > tol)

    def newton_descent(W):
        def iter(carry):
            step, W, _ = carry
            dW = compute_update_matrix(X, W)
            W += learning_rate * dW
            W, _ = jnp.linalg.qr(W)
            return step + 1, W, dW

        dW0 = jnp.zeros_like(W)
        _, W, _ = lax.while_loop(cond, iter,
                                 (0, W, dW0))
        return W

    W0 = random_semiorthogonal_matrix(op_key, shape=(n_components, n_features))
    W = newton_descent(W0)
    return W
