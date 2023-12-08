import jax
import jax.numpy as jnp


def gs(A):
    """
    Gram-Schmidt orthogonalization algorithm for non-square matrix.
    This function could be used in the FastICA algorithm, however, it would be non-optimal.
    :return: semiorthogonal matrix
    """
    n, m = A.shape

    def iter(W, component):
        w = A.at[component].get()
        w = w / jnp.linalg.norm(w)

        @jax.vmap
        def op(wj):
            return (w.T @ wj) * wj

        sum = jnp.sum(op(W), axis=0).reshape((m,))  # could be optimized..
        w = w - sum
        w = w / jnp.linalg.norm(w)
        W = W.at[component].set(w)
        return W, None

    A_gs, _ = jax.lax.scan(iter, jnp.zeros(shape=(n, m)), jnp.arange(n))
    return A_gs


def gs_sampling(op_key, shape):
    """
    Sampling random semi-orthogonal matrix by using the Gram-Schmidt orthogonalization algorithm for non-square matrix.
    Assuming A nxm with n <= m and rank n.
    :return semi-orthogonal matrix
    """
    A = jax.random.uniform(op_key, shape=shape, minval=-1, maxval=1)
    return gs(A)
