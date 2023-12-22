from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.ivae import train_and_evaluate
from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.data import DataSet
import jax.numpy as jnp
import jax
import flax.linen as nn


def main():
    OP_key = jax.random.PRNGKey(1488)

    dataset = jnp.load("scripts/VAE_toy_model/data/data.npz")
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
        'batch_size': 64,
        'epochs': 200,
    }

    mcc_scores = train_and_evaluate(OP_key, dataset, model_cfg, learning_cfg)
    # LaTeX code for the subplot
    subplot_code_ = r'''
    \nextgroupplot[
            title=Evolution of MCC during FastICA algorithm iterations on synthetic data,
            xlabel={iterations},
            ylabel={MCC Score},
            xmin=0, xmax=%d,
            ymin=%f, ymax=%f
            ]
    \addplot[
        color=blue,
        mark=*,
        style={very thick},
    ] coordinates {
        %s
    };
    ''' % (len(mcc_scores)-1, min(mcc_scores)-0.1, 1, ' '.join(f'({i},{score:.2f})' for i, score in enumerate(mcc_scores)))

    # Save the LaTeX code to a file
    file_path = 'report/figures_latex/mcc_ivae_synthetic.tex'  # Path to save the .tex file
    with open(file_path, 'w') as file:
        file.write(subplot_code_.strip())



if __name__ == "__main__":
    main()
