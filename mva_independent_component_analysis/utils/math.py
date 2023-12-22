from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=[1])
def logaddexp(arr, axis=-1):
    """
    Compute the log of the sum of exponentials of input elements.
    Similar to applying reduce to logaddexp.
    """
    max_arr = jnp.max(arr, axis=axis)
    return max_arr + jnp.log(jnp.sum(jnp.exp(arr - max_arr), axis=axis))
