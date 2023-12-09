from flax import linen as nn
import jax.numpy as jnp
import jax
from .exponential_family import logdensity_normal
from typing import Callable
from functools import partial


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
    OP_key: jax.random.PRNGKey
    data_dim: int
    latent_dim: int
    aux_dim: int
    n_layers: int = 3
    activation: Callable = lambda x: nn.leaky_relu(x, negative_slope=.1)
    hidden_dim: int = 50

    def setup(self):
        _, self.key = jax.random.split(self.OP_key, 2)

        # prior params
        self.prior_mean = jnp.zeros(1)
        self.logl = MLP(self.aux_dim, self.latent_dim, self.hidden_dim, self.n_layers, self.activation)
        # decoder params
        self.f = MLP(self.latent_dim, self.data_dim, self.hidden_dim, self.n_layers, self.activation)
        self.decoder_var = .1 * jnp.ones(1)
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
        return g, logv.exp()

    def decoder(self, s):
        f = self.f(s)
        return f

    def prior(self, u):
        logl = self.logl(u)
        return logl.exp()

    def __call__(self, key, x, u):
        l = self.prior(u)
        g, v = self.encoder(x, u)
        s = self.reparameterize(key, g, v)
        f = self.decoder(s)
        return f, g, v, s, l


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    def __init__(self):
        super().__init__()
        self.c = 2 * jnp.pi * jnp.ones(1)
        self.name = 'gauss'

    def sample(self, key, mu, var):
        eps = jax.random.normal(key, mu.size()).squeeze()
        return mu + jnp.sqrt(var) * eps

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.reshape(param_shape), v.reshape(param_shape)
        lpdf = -0.5 * (jnp.log(self.c) + jnp.log(v) + (x - mu) ** 2 / v)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = jnp.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = jnp.linalg.inv(cov)  # works on batches
        c = d * jnp.log(self.c)
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + jnp.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: jnp.array):
        """
        compute the log of the absolute value of determinants for a batch of 2D matrices. Uses jnp.slogdet
        this implementation is just a for loop, but that is what's suggested in torch forums
        gpu compatible
        """

        out = jax.lax.map(jax.lax.stop_gradient(jnp.slogdet), cov_batch)  # à check
        signs, logabsdets = out
        return signs, logabsdets


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, slope, device, fixed_mean=None,
                 fixed_var=None):
        super().__init__()
        self.distribution = Normal(device=device)
        if fixed_mean is None:
            self.mean = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        else:
            self.mean = lambda x: fixed_mean * jnp.ones(1)
        if fixed_var is None:
            self.log_var = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        else:
            self.log_var = lambda x: jnp.log(fixed_var) * jnp.ones(1)

    def sample(self, *params):
        return self.distribution.sample(*params)

    def log_pdf(self, x, *params, **kwargs):
        return self.distribution.log_pdf(x, *params, **kwargs)

    def forward(self, *input):
        if len(input) > 1:
            x = jax.lax.concatenate(input, dim=1)
        else:
            x = input.at[0].get()
        return self.mean(x), self.log_var(x).exp()
