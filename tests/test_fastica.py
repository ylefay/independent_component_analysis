import jax
import jax.numpy as jnp
from mva_independent_component_analysis.fast_ica.preprocessing import demeaning, whitening
from mva_independent_component_analysis.fast_ica.fastica import fast_ica
from mva_independent_component_analysis.fast_ica.discriminating_fastica import fast_ica as discriminating_fast_ica

import numpy.testing as npt
from scipy.signal import sawtooth
import pytest
from functools import partial

FAST_ICAs = [partial(fast_ica, fun=jnp.tanh), discriminating_fast_ica]  # test does not pass on mle_fast_ica.


@pytest.mark.parametrize("fast_ica_implementation", FAST_ICAs)
def test_fastica(fast_ica_implementation):
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

    centred_X, _ = demeaning(X)
    whitened_X, _ = whitening(X)
    W = fast_ica_implementation(op_key=JAX_KEY, X=whitened_X, n_components=X.shape[0], tol=1e-5, max_iter=1000)
    S = W.T @ whitened_X


@pytest.mark.parametrize("fast_ica_implementation", FAST_ICAs)
def test_ica_identification(fast_ica_implementation):
    # Blind source separation problem (BSS)
    JAX_KEY = jax.random.PRNGKey(1337)
    ns = jnp.linspace(0, 200, 500)
    # Sources
    S = jnp.array([jnp.sin(ns * 1),
                   sawtooth(ns * 1.9),
                   jax.random.uniform(JAX_KEY, shape=(len(ns),))])
    n_sources, n_samples = S.shape
    # Mixing process
    A = jnp.array([[0.5, 1, 0.2],
                   [1, 0.5, 0.4],
                   [0.5, 0.8, 1]])
    # Mixed signals
    X = A @ S
    # Whiten mixed signals
    X, mean = demeaning(X)
    X, whiteM = whitening(X)
    # Running ICA.
    W = fast_ica_implementation(op_key=JAX_KEY, X=X, n_components=X.shape[0], tol=1e-8, max_iter=5000)
    # Testing for orthogonality
    npt.assert_array_almost_equal(W @ W.T, jnp.identity(n_sources), decimal=2)
    # Estimated sources
    S_est = W.T @ X
    # For comparison purposes
    S, _ = demeaning(S)
    S, _ = whitening(S)
    # Finding the permutation of the signals
    perm = jnp.argmax(jnp.abs(S_est @ S.T),
                      axis=0)  # trick to find the permutation of the signals
    # Test the ratio of the reconstructed signal to the original signals
    for i in range(n_sources):
        ratio = jnp.abs(S_est[perm[i]] / S[i])
        std = jnp.std(ratio)
        mean = jnp.mean(ratio)
        ratio = ratio[(ratio - mean) / std < 2]  # removing outliers, ICA is sensitive to outliers
        npt.assert_allclose(jnp.mean(ratio), 1,
                            atol=0.2)  # in average, the reconstructed signal should be equal to the original signal up to scaling
