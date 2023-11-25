import jax.numpy as jnp
import jax


def fast_ica(op_key, X, n_components=None, tol=1e-2, fun=jnp.tanh, max_iter=10 ** 5):
    """
    FastICA algorithm for independent component analysis.
    :param X: data matrix of shape (n_features, n_samples)
    :param n_components: number of desired components
    :param tol: tolerance for the stopping criterion
    :param fun: function used for the approximation of the negentropy
    :param max_iter: maximum number of iterations for each component
    :return: matrix of shape (n_features, n_components) containing the components
    """
    n_features, n_samples = X.shape
    if n_components is None:
        n_components = n_features

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
            fun_image, fun_p_img = jax.vmap(jax.value_and_grad(fun))(w.T @ X)
            w = 1 / n_samples * X @ fun_image.T - 1 / n_samples * fun_p_img @ jnp.ones((n_samples,)) * w

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
