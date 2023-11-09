import jax
import jax.numpy as jnp
from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.generate_data.exponential_family \
    import logdensity_function_univ_normal, logdensity_function_multivariate_normal
import numpy.testing as npt


def test_sum_to_one_normal_univ():
    JAX_KEY = jax.random.PRNGKey(1337)
    mu = jax.random.uniform(JAX_KEY, (1,), minval=-5, maxval=5)
    sigma = jax.random.uniform(JAX_KEY, (1,), minval=1,
                               maxval=5)

    def log_density_function(z):
        logdensity, _, _, _ = logdensity_function_univ_normal(z, mu, sigma)
        return logdensity

    integration_linspace = jnp.linspace(-4 * sigma, 4 * sigma, 100) + mu

    @jax.vmap
    def density(z):
        return jnp.exp(log_density_function(z))

    trapz = jnp.trapz(density(integration_linspace), integration_linspace, axis=0)
    npt.assert_almost_equal(trapz, 1, decimal=2)


def test_parameters_multi_normal():
    # This tests that the mean and the variance for exponential distributions are correctly recovered
    raise NotImplementedError
