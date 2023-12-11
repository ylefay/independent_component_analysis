"""
Generate artificial data with arbitrary mixing with latent variables
Non-stationary sources are generated following a TCL distribution.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import scipy
from scipy.stats import hypsecant
from typing import Callable

from mva_independent_component_analysis.utils.mixing_matrix import generate_mixing_matrix


def generate_nonstationary_sources(OP_key, n_per_seg, n_seg, d, prior='gauss', var_lb=0.5, var_ub=3,
                                   uncentered=False, centers=None, staircase=False):
    """
    Generate non-stationary independent time sources following a TCL distribution.
    Within a segment, the distribution of each source is part of the Exponential family, and the sources
    are independent of each other.

    :param OP_key: jax.random.PRNGKey
    :param n_per_seg: int, number of data points per segment
    :param n_seg: int, number of segments
    :param d: int, number (dimension) of sources
    :param prior: str, prior distribution of the sources. One of 'lap', 'hs', 'gauss'
    :param var_lb: float, lower bound on the variance of the sources
    :param var_ub: float, upper bound on the variance of the sources
    :param uncentered: bool, if True, the sources are not centered
    :param centers: array, shape (n_seg, d), if uncentered is True, the sources are centered around these values
    :param staircase: bool, if True, s_1 will have a staircase form, used to break TCL.
    :return: tuple:
        sources: output source array of shape (n, d)
        labels: label for each point; the label is the component
        m: mean of each component
        L: modulation parameter of each component
    """
    n = n_per_seg * n_seg
    key1, key = jax.random.split(OP_key, 2)
    L = var_lb + jax.random.uniform(key=key1, shape=(n_seg, d)) * (var_ub - var_lb)
    if uncentered:
        if centers is not None:
            assert centers.shape == (n_seg, d)
            m = centers
        else:
            key1, key = jax.random.split(key, 2)
            m = -5 + 10 * jax.random.uniform(key=key1, shape=(n_seg, d))
    else:
        m = jnp.zeros((n_seg, d))

    if staircase:
        m1 = 3 * jnp.arange(n_seg).reshape((-1, 1))
        key1, key = jax.random.split(key, 2)
        a = jax.random.permutation(key=key1, x=n_seg)
        m1 = m1.at[a].get()
        if uncentered:
            key1, key = jax.random.split(key, 2)
            m2 = 2 * jax.random.uniform(key=key1, shape=(n_seg, d - 1)) - 1
        else:
            m2 = jnp.zeros((n_seg, d - 1))
        m = jnp.concatenate([m1, m2], axis=1)

    labels = jnp.zeros(n)
    key1, key = jax.random.split(key, 2)
    if prior == 'lap':
        sources = 1 / jnp.sqrt(2) * jax.random.laplace(key=key1, shape=(n, d))
    elif prior == 'hs':
        sources = 2 / jnp.pi * jnp.asarray(scipy.stats.hypsecant.rvs(0, 1, (n, d)))
    elif prior == 'gauss':
        sources = jax.random.normal(key=key1, shape=(n, d))

    def iter(carry, inps):
        sources, labels = carry
        seg, segId, L_i, m_i = inps
        sources = sources.at[segId].set(sources.at[segId].get() * L_i + m_i)
        labels = labels.at[segId].set(seg)
        return (sources, labels), None

    (sources, labels), _ = jax.lax.scan(iter, (sources, labels), (
        jnp.arange(n_seg), jnp.arange(n_seg * n_per_seg).reshape((n_seg, n_per_seg)), L, m))

    return sources, labels, m, L


def generate_data(OP_key, n_per_seg, n_seg, n_components, n_features=None, n_layers=3, prior='gauss',
                  activation=nn.leaky_relu, var_lb=0.5, var_ub=3, lin_type='uniform', n_iter_4_cond=1000, noise=0,
                  uncentered=False, centers=None, staircase=False, repeat_linearity=False):
    """
    Generate artificial data with arbitrary mixing of latent variables

    :param OP_key: jax.random.PRNGKey
    :param n_per_seg: int, number of points per segment
    :param n_seg: int, number of segments
    :param n_components: int, number (dimension) of latent sources
    :param n_features: int, number (dimension) of signals
    :param n_layers: int, number of layers in the mixing MLP
    :param prior: str, distribution of the sources. can be `lap` for Laplace , `hs` for Hypersecant or `gauss` for Gaussian
    :param activation: callable, activation function for the mixing MLP, can be `lrelu` for leaky ReLU or `xtanh` for id + tanh, or 'sigmoid'
    :param var_lb: float, lower bound for the modulation parameter
    :param var_ub: float, upper bound for the modulation parameter
    :param lin_type: str, type of linearity for the mixing MLP, can be `uniform` for uniform or `orthogonal` for normal
    :param n_iter_4_cond: int, required number of iterations for the condition number of the mixing matrix, see mixing_matrix.py
    :param noise: float, noise level
    :param uncentered: see generate_nonstationary_sources
    :param centers: see generate_nonstationary_sources
    :param staircase: see generate_nonstationary_sources
    :param repeat_linearity:
    :return: tuple of batches of generated (sources, data, auxiliary variables, mean, variance)
    """

    if n_features is None:
        n_features = n_components

    key1, key = jax.random.split(OP_key, 2)

    # sources
    S, U, M, L = generate_nonstationary_sources(key1, n_per_seg, n_seg, n_components, prior=prior,
                                                var_lb=var_lb, var_ub=var_ub,
                                                uncentered=uncentered, centers=centers, staircase=staircase)
    # non linearity
    act_f = activation

    if not repeat_linearity:
        X = S.copy()
        for nl in range(n_layers):
            key1, key = jax.random.split(key, 2)
            A = generate_mixing_matrix(key1, X.shape[1], n_features, lin_type=lin_type,
                                       n_iter_4_cond=n_iter_4_cond,
                                       staircase=staircase)
            if nl == n_layers - 1:
                X = X @ A
            else:
                X = act_f(X @ A)

    else:
        assert n_layers > 1  # suppose we always have at least 2 layers. The last layer doesn't have a non-linearity
        key1, key = jax.random.split(key, 2)
        A = generate_mixing_matrix(key1, n_components, n_features, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond,
                                   staircase=staircase)
        X = act_f(S @ A)
        if n_components != n_features:
            key1, key = jax.random.split(key, 2)
            B = generate_mixing_matrix(key1, n_features, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond,
                                       staircase=staircase)
        else:
            B = A
        for nl in range(1, n_layers):
            if nl == n_layers - 1:
                X = X @ B
            else:
                X = act_f(X @ B)

    # add noise:
    if noise:
        key1, key = jax.random.split(key, 2)
        X += noise * jax.random.normal(key1, X.shape)

    U = jax.nn.one_hot([U], num_classes=n_seg)[0]
    return S, X, U, M, L


class DataSet():
    """
    Dataset class for the generated data of the form (S=sources, X=data, U=auxiliary variables, M=mean, L=variance)
    """

    def __init__(self, data):
        self.data = data.f
        self.s = jnp.asarray(self.data.s)
        self.x = jnp.asarray(self.data.x)
        self.u = jnp.asarray(self.data.u)
        self.l = jnp.asarray(self.data.L)
        self.m = jnp.asarray(self.data.m)
        self.len = self.x.shape[0]
        self.latent_dim = self.s.shape[1]
        self.aux_dim = self.u.shape[1]
        self.data_dim = self.x.shape[1]

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim
