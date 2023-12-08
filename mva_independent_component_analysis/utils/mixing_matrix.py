import jax.numpy as jnp
import jax


def generate_mixing_matrix(OP_key, n_components: int, n_features=None, lin_type='uniform', cond_threshold=25,
                           n_iter_4_cond=None, staircase=False):
    """
    Generate mixing matrix
    :param OP_key:
    :param n_components: number of sources
    :param n_features: number of signals
    :param lin_type: uniform or orthogonal matrix
    :param cond_threshold: upper bound on the condition number of the matrix to avoid ill-posed problem
    :param n_iter_4_cond: ignore cond_treshold, compute n_iter_4_cond matrix and accept low conditioned matrix
    :param staircase: if True, generate mixing that preserves staircase form of sources
    :return: mixing matrix of shape (n_components, n_features)
    """

    key, key1 = jax.random.split(OP_key, 2)

    def _gen_matrix(key, ds, dd):
        A = (2 * jax.random.uniform(key, (ds, dd)) - 1)
        A /= jnp.linalg.norm(A, axis=0)
        return A

    def _gen_matrix_staircase(key, ds, dd, sq=None):
        if sq is None:
            sq = dd > 2
        A1 = jnp.zeros((ds, 1))  # first row of A should be e_1
        A1 = A1.at[0, 0].set(1)
        A2 = 2 * jax.random.uniform(key, (ds, dd - 1)) - 1
        if sq:
            A2 = A2.at[0].set(0)
        A = jnp.concatenate([A1, A2], axis=1)
        A /= jnp.linalg.norm(A, axis=0)
        return A

    if n_features is None:
        n_features = n_components

    if lin_type == 'orthogonal':
        A = (jnp.linalg.qr(2 * jax.random.uniform(key1, (n_components, n_features)) - 1)[0])

    elif lin_type == 'uniform':
        if n_iter_4_cond is None:
            cond_thresh = cond_threshold
        else:
            cond_list = jnp.zeros(n_iter_4_cond)
            for _ in range(n_iter_4_cond):
                key1, key = jax.random.split(key, 2)
                A = 2 * jax.random.uniform(key1, (n_components, n_features)) - 1
                A /= jnp.linalg.norm(A, axis=0)
                cond_list = cond_list.at[_].set(jnp.linalg.cond(A))
            cond_thresh = jnp.percentile(cond_list, 25)  # only accept those below 25% percentile

        gen_mat = _gen_matrix if not staircase else _gen_matrix_staircase
        key1, key = jax.random.split(key, 2)
        A = gen_mat(key1, n_components, n_features)
        while jnp.linalg.cond(A) > cond_thresh:
            key1, key = jax.random.split(key, 2)
            A = gen_mat(key1, n_components, n_features)
    return A