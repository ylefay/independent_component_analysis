import jax
import jax.numpy as jnp


@jax.vmap
def super_or_sub_gaussian(x):
    """
    See "Different Estimation Methods for the Basic Independent Component Analysis Model", Zhenyi An
    Theorem 3.1.11.

    :param x: single source of shape (n_samples, )
    :return: array of shape (n_samples, ),
        appropriate derivative of log-density for x depending on estimated sub or super-gaussianity of x.
    """

    def get_subgaussian_log_prob(x):
        """
        Subgaussian log probability of a single source x.
        Assumption:
                log(p_i(x)) = alpha_1 - 2ln(cosh(x))
                p_i = exp(alpha_1 - 2 ln(cosh(x))
        with alpha_1 = - ln(2)
        Using
            log cosh(x) = logaddexp(x, -x) - log(2)

        :param x: an array of shape (n_samples, ), float.

        """
        return -2 * (jnp.logaddexp(x, -x)) - jnp.log(2)

    def get_supergaussian_log_prob(x):
        """
        Supergaussian log probability of a single source x.
        Assumption:
            log(p_i(x)) = alpha_2 - x^2/2 + 2 * ln(cosh(x))
            p_(x) = exp(alpha_2 - x^2/2 + 2 * ln(cosh(x)))
        with alpha_2 = - ln(sqrt(2pi)) - 1/2

        :param x: an array of shape (n_samples, ), float.

        """
        return 2 * jnp.logaddexp(x, -x) - jnp.log(jnp.sqrt(2 * jnp.pi)) - 1 / 2

    def minus(x):
        g, dg = jax.value_and_grad(jax.grad(get_subgaussian_log_prob))(x)
        return x * g - dg

    return jax.lax.cond(minus(x) > 0,
                        lambda _x: jax.grad(get_subgaussian_log_prob)(x),
                        lambda _x: jax.grad(get_supergaussian_log_prob)(x),
                        x
                        )


def fast_ica(op_key, X, n_components=None, tol=1e-2, max_iter=10 ** 5):
    """
    Variation of the FastICA algorithm for independent component analysis,
    Motivated by a MLE approach.
    The log-density g_i for the i-th source, s_i, is chosen depending on the
    estimated subgaussianity or supergaussianity of s_i.

    :param X: data matrix of shape (n_features, n_samples)
    :param n_components: number of desired components
    :param tol: tolerance for the stopping criterion
    :param max_iter: maximum number of iterations for each component
    :return: matrix of shape (n_features, n_components) containing the components
    """
    N, M = X.shape
    if n_components is None:
        n_components = N

    def fun_p(x):
        return jax.vmap(jax.grad(super_or_sub_gaussian))(x)

    def cond(args):
        step, diff, _ = args
        return (step < max_iter) & (diff > tol)

    def iter_one_component(inp, key_i):
        component, W = inp
        w_init = jax.random.uniform(key_i, (N,), minval=-1, maxval=1)
        w_init = w_init / jnp.linalg.norm(w_init)

        def iter(inps):
            step, diff, w = inps
            old_w = w
            w = 1 / M * X @ super_or_sub_gaussian(w.T @ X).T - 1 / M * super_or_sub_gaussian(w.T @ X) @ jnp.ones(
                (M,)) * w

            @jax.vmap
            def op(wj):
                return (w.T @ wj) * wj

            sum = jnp.sum(op(W.T), axis=0).reshape((N,))  # could be optimized..
            w = w - sum
            w = w / jnp.linalg.norm(w)
            return step + 1, jnp.linalg.norm(w - old_w), w

        _, _, w = jax.lax.while_loop(cond, iter, (0, tol + 1, w_init))
        W = W.at[:, component].set(w.T)
        return (component + 1, W), None

    keys = jax.random.split(op_key, n_components)
    out, _ = jax.lax.scan(iter_one_component, (0, jnp.zeros((N, n_components))), keys)
    _, W = out
    return W
