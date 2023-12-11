import jax.numpy as jnp
import jax
from mva_independent_component_analysis.utils.linear_sum_assignement import linear_sum_assignment
from scipy.stats import spearmanr
from functools import partial


def ratio(S_est, S):
    """
    Find the most plausible permutation between the signal and the estimated signals.
    Compute the ratio between the paired signals and the estimated signals.
    """
    perm = jnp.argmax(jnp.abs(S_est @ S.T),
                      axis=0)
    S_est = jax.lax.permute(S_est, perm, axis=0)
    return S_est / S


def mean_corr_coef(S_est, S, method='pearson'):
    """
    Mean correlation coefficient metric.

    :param S: numpy.ndarray
    :param S_est: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
            The options are 'pearson' and 'spearman'
            'pearson':
                use Pearson's correlation coefficient
            'spearman':
                use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    d = S.shape[1]
    if method == 'pearson':
        cc = jnp.corrcoef(S, S_est, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(S, S_est)[0][:d, d:]
    cc = jnp.abs(cc)
    perm = linear_sum_assignment(-1 * cc)
    score = cc[linear_sum_assignment(-1 * cc)].mean()
    return score, perm
