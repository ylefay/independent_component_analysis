# Independent Component Analysis (ICA)

Companion code to the project done for the MVA
course [Probabilistic Graphical Models](https://lmbp.uca.fr/~latouche/mva/IntroductiontoProbabilisticGraphicalModelsMVA.html)
on Independent Component Analysis.
It includes:

- A Numpy implementation of the FastICA algorithm
- Two jax implementation of the FastICA Algorithm, one with a discriminating prior depending on the estimated (non)
  -Gaussianity of each source.
- A Jax implementation of the Gradient Descent for the maximum-likelihood estimator, with a discriminating prior
  depending on the estimated (non)-Gaussianity of each source.
- A Flax (Jax) implementation of identifiable Variational Autoencoder (iVAE) for ICA

Full report available at [here](https://github.com/ylefay/independent_component_analysis/blob/main/report/report.pdf)
## Audio Source Separation Using ICA

Here are some audio samples of the source separation results:

*Speech signals*

- Original Mixed Audio: [Listen](experiments/exp2_speech/talks_mixture.wav)
- Separated Audio Source 1: [Listen](experiments/exp2_speech/output/s3_predicted.wav)
- Separated Audio Source 2: [Listen](experiments/exp2_speech/output/s4_predicted.wav)
- Separated Audio Source 3: [Listen](experiments/exp2_speech/output/s5_predicted.wav)

*Sound signals*

- Original Mixed Audio: [Listen](experiments/exp1_sounds/sound_mixture.wav)
- Separated Audio Source 1: [Listen](experiments/exp1_sounds/output/s1_predicted.wav)
- Separated Audio Source 2: [Listen](experiments/exp1_sounds/output/s2_predicted.wav)

## References

* [Independent Component Analysis: Algorithms and Applications](https://www.sciencedirect.com/science/article/pii/S0893608000000265)
* [Variational Autoencoders and Nonlinear ICA:
  A Unifying Framework](https://proceedings.mlr.press/v108/khemakhem20a.html)

## Authors

Zineb Bentires, Nour Bouayed, Yvann Le Fay
