from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.ivae import train_and_evaluate
from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.data import DataSet
import jax.numpy as jnp
import jax
import flax.linen as nn


def main():
    OP_key = jax.random.PRNGKey(1337)

    dataset = jnp.load("./data/data.npz")
    dataset = DataSet(dataset)

    model_cfg = {
        'n_layers': 3,
        'activation': lambda x: nn.tanh(x) + 0.1 * x,
        'hidden_dim': 100}
    learning_cfg = {
        'a': 100,
        'b': 1,
        'c': 0,
        'd': 10,
        'lr': 1e-2,
        'batch_size': 10,
        'epochs': 10,
    }

    train_and_evaluate(OP_key, dataset, model_cfg, learning_cfg)


if __name__ == "__main__":
    main()
