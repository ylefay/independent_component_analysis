from .nets import IVAE
from optax import adamw
from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.data import DataSet
import jax.numpy as jnp
import jax


def create_batch(OP_key, dataset, batch_size):
    n = dataset.len // batch_size

    def split_and_shuffle(key, x):
        x = jnp.asarray(jnp.split(x, n))
        x = jax.random.permutation(key, x)
        return x

    return split_and_shuffle(OP_key, dataset.x), split_and_shuffle(OP_key, dataset.s), split_and_shuffle(OP_key, dataset.u)


def train_and_evaluate(OP_key, dataset, model_cfg, learning_cfg):
    assert isinstance(dataset, DataSet)

    data_dim, latent_dim, aux_dim = dataset.get_dims()
    model_cfg.update({'data_dim': data_dim, 'latent_dim': latent_dim, 'aux_dim': aux_dim})

    batch_size = learning_cfg.pop('batch_size', 64)
    assert dataset.len % batch_size == 0
    key1, key = jax.random.split(OP_key, 2)
    batches = create_batch(key1, dataset, batch_size)

    # learning parameters
    lr = learning_cfg.get('lr', 1e-3)
    model = IVAE(key, **model_cfg)
    optimizer = adamw(learning_rate=lr)

    def iter_train(carry, batch):
        x, u, s = batch
        return None,
    jax.lax.scan(iter_train, None, batches)