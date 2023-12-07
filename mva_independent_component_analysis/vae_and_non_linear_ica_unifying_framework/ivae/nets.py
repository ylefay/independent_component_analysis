from flax import linen as nn
import jax.numpy as jnp
import jax
from exponential_family import logdensity_normal


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        if isinstance(hidden_dim, int):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('hidden_dim must be either an int or a list of ints: {}'.format(hidden_dim))
        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.activation = activation
        else:
            raise ValueError('activation must be either a str or a list of strs: {}'.format(activation))
        self.slope = slope
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: nn.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: nn.tanh(x) + slope * x)
            elif act == 'sigmoid':
                self._act_f.append(nn.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))
        if self.n_layers == 1:
            fc_list = [nn.linear(self.input_dim, self.output_dim)]
        else:
            fc_list = [nn.linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                fc_list.append(nn.linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            fc_list.append(nn.linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
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
    def __init__(self, data_dim, latent_dim, aux_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1):
        super().__init__()
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
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s, l

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        f, g, v, z, l = self.forward(x, u)
        M, d_latent = z.size()
        logpx = logdensity_normal(x, f, self.decoder_var).sum(dim=-1)
        logqs_cux = logdensity_normal(z, g, v).sum(dim=-1)
        logps_cu = logdensity_normal(z, None, l).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
        logqs_tmp = logdensity_normal(z.view(M, 1, d_latent), g.view(1, M, d_latent), v.view(1, M, d_latent))
        logqs = jnp.logaddexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - jnp.log(M * N)
        logqs_i = (jnp.logaddexp(logqs_tmp, dim=1, keepdim=False) - jnp.log(M * N)).sum(dim=-1)

        elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps_cu)).mean()
        return elbo, z
