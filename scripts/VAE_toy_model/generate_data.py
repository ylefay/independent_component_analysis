"""
This script tries to replicate the results of the experiments of the paper:
Variational Autoencoders and Nonlinear ICA: A Unifying Framework
https://arxiv.org/pdf/1907.04809.pdf
"""

import jax
import jax.numpy as jnp

from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.data import generate_data


def main():
    OP_key = jax.random.PRNGKey(1337)

    generation_kwargs = {
        'n_per_seg': 1000,  # M in the paper
        'n_seg': 40,  # L in the paper 4000
        'n_components': 5,  # n in the paper
        'n_features': 5,  # d in the paper
        'n_layers': 3,
        'prior': 'gauss',  # k=2
        'activation': lambda x: jnp.tanh(x) + 0.1 * x,
        'noise': 0.0,  # noiseless model
        'staircase': False
    }

    path = "./data/"
    S, X, U, M, L = generate_data(OP_key=OP_key, **generation_kwargs)
    jnp.savez(path + "data.npz", s=S, x=X, u=U, m=M, L=L)


if __name__ == "__main__":
    main()
