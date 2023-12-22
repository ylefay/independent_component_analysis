import jax
import jax.numpy as jnp

from mva_independent_component_analysis.utils import centering_and_whitening, generate_mixing_matrix
from mva_independent_component_analysis.fast_ica.fastica import fast_ica
from mva_independent_component_analysis.fast_ica.discriminating_fastica import fast_ica as discriminating_fast_ica
from mva_independent_component_analysis.mle_ica.gradient import gradient_ica, subgaussian, supergaussian

import numpy.testing as npt
from scipy.signal import sawtooth
import pytest
from functools import partial

ICAs = [partial(fast_ica, fun=jnp.tanh), discriminating_fast_ica, gradient_ica, partial(gradient_ica, g=subgaussian),
        partial(gradient_ica, g=supergaussian)]  # test does not pass on mle_fast_ica.


@pytest.mark.parametrize("ica_implementation", ICAs)
def test_fastica(ica_implementation):
    """
    Test the fast_ica function.
    Just making sure it does not fail.
    """
    JAX_KEY = jax.random.PRNGKey(1337)
    n_samples = 1000
    min_features = 2
    max_features = 100

    n_features = jax.random.randint(JAX_KEY, (1,), min_features, max_features).at[0].get()
    _, key_samples = jax.random.split(JAX_KEY, 2)
    X = jax.random.normal(key_samples, (n_features, n_samples))

    X, _, _ = centering_and_whitening(X)
    n_components = X.shape[0] - 1  # hardcoding one less than n_features to test for non square matrices
    W = ica_implementation(op_key=JAX_KEY, X=X, n_components=n_components, tol=1e-5, max_iter=1000)
    S = W @ X
    npt.assert_array_almost_equal(W @ W.T, jnp.identity(W.shape[0]), decimal=2)


ICAs = [partial(fast_ica, fun=jnp.tanh), discriminating_fast_ica,
        gradient_ica]  # still need to create a test for sub super gaussian priors


@pytest.mark.parametrize("ica_implementation", ICAs)
def test_ica_identification(ica_implementation):
    # Blind source separation problem (BSS)
    JAX_KEY = jax.random.PRNGKey(1337)
    ns = jnp.linspace(0, 200, 500)
    # Sources
    S = jnp.array([jnp.sin(ns * 1),
                   sawtooth(ns * 1.9),
                   jax.random.uniform(JAX_KEY, shape=(len(ns),))])
    n_sources, n_samples = S.shape
    # Mixing process
    A = generate_mixing_matrix(JAX_KEY, n_sources, n_sources + 1, n_iter_4_cond=None).T
    # Mixed signals
    X = A @ S
    # Whiten mixed signals
    X, meanX, whiteM = centering_and_whitening(X)
    # Running ICA
    if gradient_ica == ica_implementation:
        max_iter = 500000  # gradient ica is slower
    else:
        max_iter = 5000
    W = ica_implementation(op_key=JAX_KEY, X=X, n_components=n_sources, tol=1e-8, max_iter=max_iter)
    # Testing for orthogonality
    npt.assert_array_almost_equal(W @ W.T, jnp.identity(n_sources), decimal=2)
    # Estimated sources
    S_est = W @ X
    # For comparison purposes
    S, meanS, whiteS = centering_and_whitening(S)
    # Finding the permutation of the signals
    perm = jnp.argmax(jnp.abs(S_est @ S.T),
                      axis=0)  # trick to find the permutation of the signals
    # Test the ratio of the reconstructed signal to the original signals
    for i in range(n_sources):
        ratio = jnp.abs(S_est[perm[i]] / S[i])
        std = jnp.std(ratio)
        mean = jnp.mean(ratio)
        ratio = ratio[(ratio - mean) / std < 1.25]
        mean = jnp.mean(ratio)
        std = jnp.std(ratio)  # very few but large outliers : over-estimated std
        ratio = ratio[(ratio - mean) / std < 1.25]
        npt.assert_allclose(jnp.mean(ratio), 1,
                            atol=0.1)  # in average, the reconstructed signal should be equal to the original signal up to scaling
