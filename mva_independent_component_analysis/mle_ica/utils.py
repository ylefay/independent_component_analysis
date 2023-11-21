import jax
import jax.numpy as jnp


def get_subgaussian_log_prob(x):
    """Subgaussian log probability of a single source x.
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
    """Supergaussian log probability of a single source x.
    Assumption:
        log(p_i(x)) = alpha_2 - x^2/2 + log(cosh(x))
        p_(x) = exp(alpha_2 - x^2/2 + log(cosh(x)))
    with alpha_2 = - ln(sqrt(2pi)) - 1/2

    :param x: an array of shape (n_samples, ), float.

    """
    return jnp.logaddexp(x, -x) - jnp.log(jnp.sqrt(2 * jnp.pi)) - 1 / 2


def super_or_sub_gaussian(x):
    """
    See "Different Estimation Methods for the Basic Independent Component Analysis Model", Zhenyi An
    Theorem 3.1.11.

    :param x: single source of shape (n_samples, )
    :return: array of shape (n_samples, ),
        appropriate derivative of log-density for x depending on estimated sub or super-gaussianity of x.
    """

    def minus(x):
        g, dg = jax.value_and_grad(jax.grad(get_subgaussian_log_prob))(x)
        return x * g - dg

    return jax.lax.cond(minus(x) > 0,
                        lambda _x: jax.grad(get_subgaussian_log_prob)(x),
                        lambda _x: jax.grad(get_supergaussian_log_prob)(x),
                        x
                        )
