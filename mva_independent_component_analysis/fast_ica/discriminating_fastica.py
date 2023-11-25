import jax
import jax.numpy as jnp


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


def fast_ica(op_key, X, n_components=None, tol=1e-2, max_iter=10 ** 5):
    """
    Variation of the FastICA algorithm for independent component analysis.
    The log-density g_i for the i-th source, s_i, is chosen depending on the
    estimated subgaussianity or supergaussianity of s_i.

    :param X: data matrix of shape (n_features, n_samples)
    :param n_components: number of desired components
    :param tol: tolerance for the stopping criterion
    :param max_iter: maximum number of iterations for each component
    :return: matrix of shape (n_features, n_components) containing the components
    """
    n_features, n_samples = X.shape
    if n_components is None:
        n_components = n_features

    def fun_funp(x):
        """
        See "Different Estimation Methods for the Basic Independent Component Analysis Model", Zhenyi An
        Theorem 3.1.11.

        :param x: single source of shape (n_samples, )
        :return: array of shape (n_samples, ),
            appropriate derivative of log-density for x depending on estimated sub or super-gaussianity of x.
        """

        def minus(x):
            # g, dg = jax.value_and_grad(subgaussian)(x)
            # another criterion is x * g - dg > 0
            return jnp.mean(x ** 4) - 3  # we use the kurtosis to discriminate.

        return jax.lax.cond(minus(x) > 0,
                            jax.vmap(jax.value_and_grad(subgaussian)),
                            jax.vmap(jax.value_and_grad(supergaussian)),
                            x
                            )

    def cond(args):
        step, diff, _ = args
        return (step < max_iter) & (diff > tol)

    def iter_one_component(inp, key_i):
        component, W = inp
        w_init = jax.random.uniform(key_i, (n_features,), minval=-1, maxval=1)
        w_init = w_init / jnp.linalg.norm(w_init)

        def iter(inps):
            step, diff, w = inps
            old_w = w
            fun_img, fun_p_img = fun_funp(w.T @ X)
            w = 1 / n_samples * X @ fun_img.T - 1 / n_samples * fun_p_img @ jnp.ones(
                (n_samples,)) * w

            @jax.vmap
            def op(wj):
                return (w.T @ wj) * wj

            sum = jnp.sum(op(W.T), axis=0).reshape((n_features,))  # could be optimized..
            w = w - sum
            w = w / jnp.linalg.norm(w)
            return step + 1, jnp.linalg.norm(w - old_w), w

        _, _, w = jax.lax.while_loop(cond, iter, (0, tol + 1, w_init))
        W = W.at[:, component].set(w.T)
        return (component + 1, W), None

    keys = jax.random.split(op_key, n_components)
    out, _ = jax.lax.scan(iter_one_component, (0, jnp.zeros((n_features, n_components))), keys)
    _, W = out
    return W
