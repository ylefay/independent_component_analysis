"""
This script tries to replicate the results of the experiments of the paper:
Variational Autoencoders and Nonlinear ICA: A Unifying Framework
https://arxiv.org/pdf/1907.04809.pdf
"""

from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.data import generate_data
import jax.numpy as jnp
import jax


def main():
    OP_key = jax.random.PRNGKey(1337)

    generation_kwargs = {
        'n_per_seg': 40,  # M in the paper
        'n_seg': 10,  # L in the paper 4000
        'n_components': 10,  # n in the paper
        'n_features': 40,  # d in the paper
        'prior': 'gauss',  # k=2
        'activation': 'lrelu',
        'noisy': 0.0,  # noiseless model
        'staircase': False
    }

    path = "./data/"
    S, X, U, M, L = generate_data(OP_key=OP_key, **generation_kwargs)
    jnp.savez(path + "data.npz", S=S, X=X, U=U, M=M, L=L)


if __name__ == "__main__":
    main()
