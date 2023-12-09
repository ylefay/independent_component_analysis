from .nets import IVAE
from optax import adamw, apply_updates
from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.data import DataSet
import jax.numpy as jnp
import jax
from .exponential_family import logdensity_normal


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

    batch_size = learning_cfg.pop('batch_size', 64)
    assert dataset.len % batch_size == 0
    size = dataset.len // batch_size
    key1, key = jax.random.split(OP_key, 2)
    batches = create_batch(key1, dataset, batch_size)

    # learning parameters
    lr = learning_cfg.get('lr', 1e-3)

    key1, key2 = jax.random.split(key, 2)

    model = IVAE(**model_cfg)
    params = model.init(key2, key=key1, x=jnp.empty((10, 40)), u=jnp.empty((10, 40)))
    optimizer = adamw(learning_rate=lr)
    opt_state = optimizer.init(params)

    def train_step(state, z_rng, x, u, N, a, b, c, d):
        def loss_fn(params):
            f, g, v, z, l = model.apply(
                {'params': params}, z_rng, x, u
            )

            M, d_latent = z.size()
            logpx = logdensity_normal(x, f, model.decoder_var).sum(dim=-1)
            logqs_cux = logdensity_normal(z, g, v).sum(dim=-1)
            logps_cu = logdensity_normal(z, None, l).sum(dim=-1)

            logqs_tmp = logdensity_normal(z.reshape(M, 1, d_latent), g.reshape(1, M, d_latent),
                                          v.reshape(1, M, d_latent))
            logqs = jnp.logaddexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - jnp.log(M * N)
            logqs_i = (jnp.logaddexp(logqs_tmp, dim=1, keepdim=False) - jnp.log(M * N)).sum(dim=-1)

            elbo = -jnp.mean((a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps_cu)))
            # return elbo, z

            return elbo

        grads = jax.grad(loss_fn, has_aux="True")(state.params)
        return state.apply_gradients(grads=grads)

    for i in range(size):
        key1, key = jax.random.split(key, 2)
        x, s, u = batches[0][i], batches[1][i], batches[2][i]
        elbo = train_step(state, key1, x, u, dataset.len, learning_cfg['a'], learning_cfg['b'], learning_cfg['c'],
                          learning_cfg['d'])
