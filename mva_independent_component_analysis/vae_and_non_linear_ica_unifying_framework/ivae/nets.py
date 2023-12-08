from flax import linen as nn
import jax.numpy as jnp
import jax
from .exponential_family import logdensity_normal
from typing import Union, Callable
from functools import partial


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        if isinstance(hidden_dim, int):
            self.hidden_dim = hidden_dim * jnp.ones(self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        if isinstance(activation, Union[str, callable]):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.activation = activation
        self.slope = slope
        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: nn.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: nn.tanh(x) + slope * x)
            elif act == 'sigmoid':
                self._act_f.append(nn.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            elif isinstance(act, Callable):
                self._act_f.append(act)
        if self.n_layers == 1:
            fc_list = [nn.linear.Dense(self.input_dim, self.output_dim)]
        else:
            fc_list = [nn.linear.Dense(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                fc_list.append(nn.linear.Dense(self.hidden_dim[i - 1], self.hidden_dim[i]))
            fc_list.append(nn.linear.Dense(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = fc_list

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


class IVAE(nn.Module):
    def __init__(self, OP_key, data_dim, latent_dim, aux_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1):
        super().__init__()
        self.OP_key = OP_key
        _, self.key = jax.random.split(OP_key, 2)
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior params
        self.prior_mean = jnp.zeros(1)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_var = .1 * jnp.ones(1)
        # encoder params
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)

    @staticmethod
    def reparameterize(key, mean, var):
        eps = jax.random.normal(key=key, shape=var.shape)
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

    def forward(self, x, u):
        l = self.prior(u)
        g, v = self.encoder(x, u)
        key1, self.key = jax.random.split(self.key, 2)
        s = self.reparameterize(key1, g, v)
        f = self.decoder(s)
        return f, g, v, s, l

    @partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        f, g, v, z, l = self.forward(x, u)
        M, d_latent = z.size()
        logpx = logdensity_normal(x, f, self.decoder_var).sum(dim=-1)
        logqs_cux = logdensity_normal(z, g, v).sum(dim=-1)
        logps_cu = logdensity_normal(z, None, l).sum(dim=-1)

        logqs_tmp = logdensity_normal(z.reshape(M, 1, d_latent), g.reshape(1, M, d_latent), v.reshape(1, M, d_latent))
        logqs = jnp.logaddexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - jnp.log(M * N)
        logqs_i = (jnp.logaddexp(logqs_tmp, dim=1, keepdim=False) - jnp.log(M * N)).sum(dim=-1)

        elbo = -jnp.mean((a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps_cu)))
        return elbo, z


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
