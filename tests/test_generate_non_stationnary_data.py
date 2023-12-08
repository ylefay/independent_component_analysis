import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy.testing as npt
from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.data import generate_data, \
    generate_nonstationary_sources
import pytest

PRIORS = ['lap', 'hs', 'gauss']
ACTIVATIONS = ['lrelu', 'sigmoid', 'xtanh', 'none', lambda x: nn.relu(x)]
LIN_TYPES = ['uniform', 'orthogonal']


@pytest.mark.parametrize("prior", PRIORS)
def test_generate_non_stationnary_data(prior):
    # This test the generate_non_stationnary_data function
    # See figure 9. in
    # Variational Autoencoders and Nonlinear ICA: A Unifying Framework
    JAX_KEY = jax.random.PRNGKey(1337)
    n_seg = 40  # number of segments
    n_per_seg = 4000  # number of pts inside each segment
    d = 10  # number of latent sources
    sources, labels, m, L = generate_nonstationary_sources(OP_key=JAX_KEY, n_per_seg=n_per_seg, n_seg=n_seg, d=d,
                                                           prior=prior, uncentered=True)
    npt.assert_allclose(jnp.mean((sources.reshape(n_seg, n_per_seg, 10)), axis=-2), m, atol=0.2)
    npt.assert_allclose(jnp.std((sources.reshape(n_seg, n_per_seg, 10)), axis=-2), L, rtol=1e-1)


@pytest.mark.parametrize("prior", PRIORS)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("lin_type", LIN_TYPES)
def test_generate_data(prior, activation, lin_type):
    # This test the function generate_data passes. No specific test is otherwise, to do.
    # See figure 9. in
    # Variational Autoencoders and Nonlinear ICA: A Unifying Framework
    JAX_KEY = jax.random.PRNGKey(1337)
    n_seg = 40  # number of segments
    n_per_seg = 10  # number of pts inside each segment
    n_components = 3  # number of latent sources
    n_features = 5  # number of features
    _ = generate_data(JAX_KEY, n_per_seg=n_per_seg, n_seg=n_seg, n_components=n_components,
                      n_features=n_features, n_layers=3,
                      prior=prior,
                      activation=activation, slope=.1, var_lb=0.5, var_ub=3, lin_type=lin_type,
                      n_iter_4_cond=5, noisy=1)
    pass