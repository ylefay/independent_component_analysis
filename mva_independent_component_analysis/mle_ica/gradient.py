import jax
import jax.numpy as jnp

from .schmidt import gs_sampling, gs


def subgaussian(x):
    """
    Subgaussian log probability of a single source x.
    Assumption:
            p_i propto Exp[-x^2/2 + Log[Cosh[x]]]
    i.e.,  g_i = x - tanh(x)
    :param x: an array of shape (n_samples, ), float.

    """
    return -jnp.tanh(x) + x


def supergaussian(x):
    """
    Supergaussian log probability of a single source x.
    Assumption:
        p_i propto Exp[-x^2/2 - Log[Cosh[x]]]
    i.e.,  g_i = x + tanh(x)
    :param x: an array of shape (n_samples, ), float.

    """
    return x + jnp.tanh(x)


def switching_criterion_kurtosis(y):
    return jnp.mean(y ** 4) - 3


def gradient_ica(op_key, X, n_components=None, tol=1e-8, max_iter=10 ** 5,
                 learning_rate=0.0001, g=None):
    """
    Perform a Gradient descent on the log-likelihood of the ICA model.
    :param op_key:
    :param X: observed signals of shape (n_features, n_samples)
    :param n_components: number of desired components / sources
    :param tol: tolerance for the Gradient's step size, used as a stopping criterion
    :param max_iter: maximum number of iterations, used as a stopping criterion
    :param learning_rate: learning rate for the Newton's descent
    :param g: given g, the prior on the log-density of the source.
    If g is none, we use the kurtosis as a criterion to discriminate between sub and supergaussian sources,
    with supergaussian and subgaussian functions previously defined.
    :return: W, the estimated mixing matrix of shape (n_components, n_features)

    Author: Yvann Le Fay & Zineb Bentires
    """
    n_features, n_samples = X.shape
    if n_components is None:
        n_components = n_features

    def compute_update_matrix(W):
        S = W @ X
        id = jnp.eye(n_components, n_components)
        if g is None:
            sign_matrix = jnp.diag(jax.vmap(switching_criterion_kurtosis)(S))
            if n_components == n_features:
                dW = (id - 1 / n_samples * (sign_matrix @ jnp.tanh(S) + S) @ S.T) @ W
            else:
                dW = jnp.linalg.inv(W @ W.T) @ W - 1 / n_samples * (sign_matrix @ jnp.tanh(S) + S) @ X.T
        else:
            if n_components == n_features:
                dW = (id - 1 / n_samples * g(S) @ S.T) @ W
            else:
                dW = jnp.linalg.inv(W @ W.T) @ W - 1 / n_samples * g(S) @ X.T
        return dW

    def cond(args):
        step, _, dW = args
        return (step < max_iter) & (jnp.linalg.norm(dW, 'fro') > tol)

    def iter(carry):
        step, W, _ = carry
        dW = compute_update_matrix(W)
        W += learning_rate * dW
        W = gs(W)
        return step + 1, W, dW

    W = gs_sampling(op_key, (n_components, n_features))
    step, W, _ = jax.lax.while_loop(cond, iter, (0, W, 2 * tol * jnp.ones((n_components, n_features))))
    return W
