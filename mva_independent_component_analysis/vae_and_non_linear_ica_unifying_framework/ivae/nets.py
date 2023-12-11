from flax import linen as nn
import jax.numpy as jnp
import jax
from typing import Callable


class MLP(nn.Module):
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

    def __call__(self, key, x, u):
        l = self.prior(u)
        g, v = self.encoder(x, u)
        s = self.reparameterize(key, g, v)
        f = self.decoder(s)
        return f, g, v, s, l
