import jax
import jax.numpy as jnp
from mva_independent_component_analysis.fast_ica.preprocessing import demeaning, whitening
from mva_independent_component_analysis.fast_ica.fastica import fast_ica
import numpy.testing as npt

def test_fastica():
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

    W = fast_ica(whitened_X, X.shape[0], 1e-5, jnp.tanh, 1000)
    S = W.T @ whitened_X


