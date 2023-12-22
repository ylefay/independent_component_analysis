import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

from mva_independent_component_analysis.utils.linear_sum_assignement import linear_sum_assignment


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

def pearson_correlation_coefficient(x, y):
    # Calculate the mean of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate the numerator and denominator for the correlation coefficient formula
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator_x = np.sqrt(np.sum((x - mean_x) ** 2))
    denominator_y = np.sqrt(np.sum((y - mean_y) ** 2))

    # Calculate the correlation coefficient
    corr_coef = numerator / (denominator_x * denominator_y)

    return corr_coef

def fast_corr_coef(S_est, S):
    '''
    Greedy linear sum assignment
    '''
    assert S_est.shape[0]==S.shape[0]
    n_signals=S.shape[0]
    correlation_matrix = np.zeros((n_signals, n_signals))
    # Compute the correlation matrix
    for i in range(n_signals):
        for j in range(n_signals):
            correlation_matrix[i, j]= np.abs(pearson_correlation_coefficient(S[i], S_est[j]))
    corr_coeffs=[]
    visited_indices=[]    
    for row in correlation_matrix:
        # Create a boolean mask to mark indices for exclusion
        mask = np.ones(row.shape, dtype=bool)
        mask[visited_indices] = False

        # Use the mask to create a sliced numpy array
        sliced_row = row[mask]
        
        # Find the maximum correlation coefficient index in the sliced row
        corr_coeff_index = np.argmax(sliced_row)
        
        # Translate the index back to the original matrix
        full_index = np.where(mask)[0][corr_coeff_index]
        
        corr_coeff_full_index = full_index
        visited_indices.append(corr_coeff_full_index)

        max_corr_value = sliced_row[corr_coeff_index]
        corr_coeffs.append(max_corr_value)

    return np.mean(corr_coeffs)
