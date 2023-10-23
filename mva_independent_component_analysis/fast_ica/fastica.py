import jax.numpy as jnp
import jax

JAX_KEY = jax.random.PRNGKey(1337)


def fast_ica(X, n_components, tol=10 ** -2, fun=jnp.tanh, max_iter=10 ** 5):
    """
    FastICA algorithm for independent component analysis.
    :param X: data matrix of shape (n_features, n_samples)
    :param n_components: number of desired components
    :param tol: tolerance for the stopping criterion
    :param fun: function used for the approximation of the negentropy
    :param max_iter: maximum number of iterations for each component
    :return: matrix of shape (n_features, n_components) containing the components
    """
    N, M = X.shape

    def fun_p(x):
        return jax.vmap(jax.grad(fun))(x)

    def cond(args):
        step, diff, _ = args
        return (step < max_iter) & (diff > tol)

    def iter_one_component(inp, key):
        component, W = inp
        w_init = jax.random.uniform(key, (N,), minval=-1, maxval=1)
        w_init = w_init / jnp.linalg.norm(w_init)

        def iter(inps):
            step, diff, w = inps
            old_w = w
            w = 1 / M * X @ fun(w.T @ X).T - 1 / M * fun_p(w.T @ X) @ jnp.ones((M,)) * w

            @jax.vmap
            def summand(wj):
                return (w.T @ wj) * wj

            sum = jnp.sum(summand(W.T), axis=0).reshape((N,)) #could be optimized..
            w = w - sum
            w = w / jnp.linalg.norm(w)
            return step + 1, jnp.linalg.norm(w - old_w), w

        _, _, w = jax.lax.while_loop(cond, iter, (0, tol + 1, w_init))
        W = W.at[:, component].set(w.T)
        return (component + 1, W), W

    keys = jax.random.split(JAX_KEY, n_components)
    out, _ = jax.lax.scan(iter_one_component, (0, jnp.zeros((N, n_components))), keys)
    _, W = out
    return W
