import optax

from .nets import IVAE
from optax import adamw, apply_updates
from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.data import DataSet
import jax.numpy as jnp
import jax
from .exponential_family import logdensity_normal
import flax


def create_batch(OP_key, dataset, batch_size):
    n = dataset.len // batch_size

    def split_and_shuffle(key, x):
        x = jnp.asarray(jnp.split(x, n))
        x = jax.random.permutation(key, x)
        return x

    return split_and_shuffle(OP_key, dataset.x), split_and_shuffle(OP_key, dataset.s), split_and_shuffle(OP_key,
                                                                                                         dataset.u)


def train_and_evaluate(OP_key, dataset, model_cfg, learning_cfg):
    assert isinstance(dataset, DataSet)

    data_dim, latent_dim, aux_dim = dataset.get_dims()
    model_cfg.update({'data_dim': data_dim, 'latent_dim': latent_dim, 'aux_dim': aux_dim})

    # learning parameters
    lr = learning_cfg.pop('lr', 1e-3)
    batch_size = learning_cfg.pop('batch_size', 64)
    epochs = learning_cfg.pop('epochs', 100)
    # fixed noise sigma^2Id
    var_decoder = 0.01

    N = dataset.len
    assert N % batch_size == 0
    n = N // batch_size
    key1, key = jax.random.split(OP_key, 2)
    batches = create_batch(key1, dataset, batch_size)

    x_shape, u_shape = batches[0][0].shape, batches[1][0].shape

    model = IVAE(**model_cfg)
    variables = model.init(key, key=key1, x=jnp.empty(x_shape), u=jnp.empty(u_shape))
    state, params = flax.core.pop(variables, 'params')
    del variables
    optimizer = adamw(learning_rate=lr)
    opt_state = optimizer.init(params)
    batch_keys = jax.random.split(key, n)

    def train_step(state, opt_state, params, z_rng_batch, x_batch, u_batch, a, b, c, d):
        def batch_loss(params):
            def loss_fn(z_rng, x, u):
                f, g, v, z, l = model.apply(
                    {'params': params, **state}, z_rng, x, u
                )
                M, d_latent = z.shape
                logpx = jnp.sum(logdensity_normal(x, f, var_decoder), axis=-1)
                logqs_cux = jnp.sum(logdensity_normal(z, g, v), axis=-1)
                logps_cu = jnp.sum(logdensity_normal(z, 0., l), axis=-1)

                logqs_tmp = logdensity_normal(z.reshape(M, 1, d_latent), g.reshape(1, M, d_latent),
                                              v.reshape(1, M, d_latent))
                logqs = jnp.logaddexp(jnp.sum(logqs_tmp, axis=-1), dim=1, keepdim=False) - jnp.log(M * N)
                logqs_i = jnp.sum(jnp.logaddexp(logqs_tmp, dim=1, keepdim=False) - jnp.log(M * N), axis=-1)

                elbo = -jnp.mean(
                    (a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps_cu)))
                return elbo, z

            loss, updated_state = jax.vmap(loss_fn, in_axes=(0, 0, 0))(z_rng_batch, x_batch, u_batch)
            return loss.mean(), updated_state

        (loss, updated_state), grads = jax.value_and_grad(
            batch_loss, has_aux=True
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, params, updated_state, loss

    for _ in range(epochs):
        opt_state, params, state, loss = train_step(state, opt_state, params, batch_keys, batches[0], batches[1],
                                                    **learning_cfg)
