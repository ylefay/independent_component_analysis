from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    """
    Standard multi-layer perceptron.
    See: https://huggingface.co/flax-community/NeuralODE_SDE/blame/955a729c0c2041e2bae8c4b3a41e3dea922bda14/models/mlp.py
    """
    input_dim: int
    output_dim: int
    hidden_dim: int
    n_layers: int
    activation: Callable = 'none'

    def setup(self):
        if self.n_layers == 1:
            self.layers = [nn.linear.Dense(self.output_dim)]
        else:
            self.layers = [nn.linear.Dense(self.hidden_dim) for i in range(self.n_layers - 1)]
            self.layers += tuple([nn.linear.Dense(self.output_dim)])

    def __call__(self, x):
        h = x
        for i, lyr in enumerate(self.layers):
            h = lyr(h)
            if i < self.n_layers - 1:
                h = self.activation(h)
        return h


class IVAE(nn.Module):
    """
    IVAE model, implementation built upon https://github.com/ilkhem/iVAE/blob/master/models/nets.py.

    Author: Yvann Le Fay
    """
    data_dim: int
    latent_dim: int
    aux_dim: int
    n_layers: int = 3
    activation: Callable = lambda x: nn.leaky_relu(x, negative_slope=.1)
    hidden_dim: int = 50

    def setup(self):
        # prior params
        self.prior_mean = jnp.zeros(1)
        self.logl = MLP(self.aux_dim, self.latent_dim, self.hidden_dim, self.n_layers, self.activation)
        # decoder params
        self.f = MLP(self.latent_dim, self.data_dim, self.hidden_dim, self.n_layers, self.activation)
        # encoder params
        self.g = MLP(self.data_dim + self.aux_dim, self.latent_dim, self.hidden_dim, self.n_layers, self.activation)
        self.logv = MLP(self.data_dim + self.aux_dim, self.latent_dim, self.hidden_dim, self.n_layers, self.activation)

    @staticmethod
    def reparameterize(key, mean, var):
        eps = jax.random.normal(key=key, shape=mean.shape)
        std = jnp.sqrt(var)
        return mean + std * eps

    def encoder(self, x, u):
        xu = jax.lax.concatenate((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, jnp.exp(logv)

    def decoder(self, s):
        f = self.f(s)
        return f

    def prior(self, u):
        logl = self.logl(u)
        return jnp.exp(logl)

    def __call__(self, key, x, u, z=None, decoder=False):
        if decoder:
            f = self.decoder(z)
            return f
        l = self.prior(u)

        g, v = self.encoder(x, u)
        s = self.reparameterize(key, g, v)
        f = self.decoder(s)
        return f, g, v, s, l
