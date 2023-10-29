import jax
import jax.numpy as jnp
import blackjax


def logdensity_function(z, u, Q, T, L):
    """
    Log-density function of the exponential family.
    :param z: latent variable
    :param u: auxiliary variable
    :param Q: base measures
    :param T: sufficient statistics of the exponential family
    :param L: corresponding parameters
    :return: log-density
    """

    @jax.vmap
    def Q_single_component(i):
        return Q.at[i].get()(z.at[i].get())

    @jax.vmap
    def T_single_component(i):
        return T.at[i].get()(z.at[i].get())

    return jnp.sum(jnp.log(Q_single_component(jnp.arange(z.shape[0])))) \
           + T_single_component(jnp.arange(z.shape[0])) @ L(u).T


def logdensity_normal_function():
    raise NotImplementedError
