import jax
import jax.numpy as jnp
from functools import partial
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


def newton_ica(op_key, X, n_components=None, tol=1e-8, max_iter=10 ** 5,
               learning_rate=0.0001, prior=None):
    n_features, n_samples = X.shape
    if n_components is None:
        n_components = n_features

    def compute_update_matrix(X, W):
        S = W @ X
        id = jnp.eye(n_components, n_components)
        if prior is None:
            sign_matrix = jnp.diag(jax.vmap(switching_criterion_kurtosis)(S))
            dW = (id - 1 / n_samples * (sign_matrix @ jnp.tanh(S) + S) @ S.T) @ W
        if prior == 'super':
            dW = (id - 1 / n_samples * jnp.sum(supergaussian(S), axis=1)) @ W
        if prior == 'sub':
            dW = (id - 1 / n_samples * jnp.sum(subgaussian(S), axis=1)) @ W
        return dW

    def cond(args):
        step, _, dW = args
        return (step < max_iter) & (jnp.linalg.norm(dW, 'fro') > tol)

    def iter(carry):
        step, W, _ = carry
        dW = compute_update_matrix(X, W)
        W += learning_rate * dW
        W = gs(W)
        return step + 1, W, dW

    W = gs_sampling(op_key, (n_components, n_features))
    step, W, _ = jax.lax.while_loop(cond, iter, (0, W, 2 * tol * jnp.ones((n_components, n_features))))
    return W
