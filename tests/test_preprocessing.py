import jax
import jax.numpy as jnp
from mva_independent_component_analysis.fast_ica.preprocessing import demeaning, whitening
import numpy.testing as npt


def test_whitened_data():
    JAX_KEY = jax.random.PRNGKey(1337)
    n_samples = 1000
    min_features = 2
    max_features = 100
    min_mean = -10
    max_mean = 10
    min_std = 0.5
    max_std = 20

    n_features = jax.random.randint(JAX_KEY, (1,), min_features, max_features).at[0].get()
    mean = jax.random.uniform(JAX_KEY, (n_features, 1), minval=min_mean, maxval=max_mean)
    std = jax.random.uniform(JAX_KEY, (n_features, 10 * n_features), minval=jnp.sqrt(min_std),
                             maxval=jnp.sqrt(max_std))
    std = std @ std.T / (jnp.sqrt(10) * n_features)

    _, key_samples = jax.random.split(JAX_KEY, 2)
    X = mean + std @ jax.random.normal(key_samples, (n_features, n_samples))

    centred_X, _ = demeaning(X)
    whitened_X, _ = whitening(X)
    npt.assert_array_almost_equal(jnp.cov(whitened_X), jnp.identity(n_features), decimal=2)