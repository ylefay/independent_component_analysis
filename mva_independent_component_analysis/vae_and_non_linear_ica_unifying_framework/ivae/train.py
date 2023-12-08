from .nets import IVAE
from optax import adamw
from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.data import DataSet


def create_batch(dataset, batch_size):
    raise NotImplementedError


def train(OP_key, dataset, model_cfg, learning_cfg):
    assert isinstance(dataset, DataSet)

    data_dim, latent_dim, aux_dim = dataset.get_dims()
    model_cfg.update({'data_dim': data_dim, 'latent_dim': latent_dim, 'aux_dim': aux_dim})

    # learning parameters
    lr = learning_cfg.get('lr', 1e-3)
    model = IVAE(OP_key, **model_cfg)
    optimizer = adamw(learning_rate=lr)
