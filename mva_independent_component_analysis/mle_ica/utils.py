import jax
import jax.numpy as jnp
import tqdm
import math

def get_subgaussian_log_prob(x):
    """Subgaussian log probability of a single source x.
    We have log(p_i(x)) = alpha_1 - 2ln(cosh(x))
            p_i = exp(alpha_1 - 2 ln(cosh(x))
        this is a density iff alpha_1 = - ln(2)
    We note that :
    log cosh(x) = log ( (exp(x) + exp(-x)) / 2 )
                = log (exp(x) + exp(-x)) - log(2)
                = logaddexp(x, -x) - log(2)
    Args
        x [source_dim]

    Returns []
    """
    return -2*(jnp.logaddexp(x,-x))-math.log(2)

def get_supergaussian_log_prob(x):
    """Supergaussian log probability of a single source x.
    log(p_i(x)) = alpha_2 - x^2/2 + log(cosh(x))
    p_(x) = exp(alpha_2 - x^2/2 + log(cosh(x)))
    hence : 
        alpha_2 = - ln(sqrt(2pi)) - 1/2
                   

    Args
        x [source_dim]

    Returns []
    """
    return jnp.logaddexp(x, -x) - math.log(math.sqrt(2*math.pi)) - 1/2

def derivative_element_wise_function(f):
    return(jax.vmap(jax.grad(f)))

def second_derivative_element_wise_function(f):
    f_second = jax.vmap(jax.grad(jax.grad(f)))
    return(f_second)

def check_super_sub_gaussian(x): 
    """Check if the single signal source verify the sub or supergaussian identity 
    E(xg(x)-g'(x)) > 0 
    where g : derivative of sub or super gaussian log probabilities

    Args:
        x (array): single source in T points
    """
    g_minus = derivative_element_wise_function(get_subgaussian_log_prob)
    g_minus_derivative = second_derivative_element_wise_function(get_subgaussian_log_prob)
    g_plus = derivative_element_wise_function(get_supergaussian_log_prob)
    g_plus_derivative = second_derivative_element_wise_function(get_supergaussian_log_prob)
    y_minus= x*g_minus(x) -   g_minus_derivative(x)
    y_plus = x*g_plus(x) - g_plus_derivative(x)

    if bool(jnp.mean(y_minus)>0.):
        return(get_subgaussian_log_prob)
    elif bool(jnp.mean(y_plus)>0.):
        return(get_supergaussian_log_prob)
    else: 
        return(None)
    

import jax
import jax.numpy as jnp
import tqdm
import math

def get_subgaussian_log_prob(x):
    """Subgaussian log probability of a single source x.
    We have log(p_i(x)) = alpha_1 - 2ln(cosh(x))
            p_i = exp(alpha_1 - 2 ln(cosh(x))
        this is a density iff alpha_1 = - ln(2)
    We note that :
    log cosh(x) = log ( (exp(x) + exp(-x)) / 2 )
                = log (exp(x) + exp(-x)) - log(2)
                = logaddexp(x, -x) - log(2)
    Args
        x [source_dim]

    Returns []
    """
    return -2*(jnp.logaddexp(x,-x))-math.log(2)

def get_supergaussian_log_prob(x):
    """Supergaussian log probability of a single source x.
    log(p_i(x)) = alpha_2 - x^2/2 + log(cosh(x))
    p_(x) = exp(alpha_2 - x^2/2 + log(cosh(x)))
    hence : 
        alpha_2 = - ln(sqrt(2pi)) - 1/2
                   

    Args
        x [source_dim]

    Returns []
    """
    return jnp.logaddexp(x, -x) - math.log(math.sqrt(2*math.pi)) - 1/2

def derivative_element_wise_function(f):
    return(jax.vmap(jax.grad(f)))

def second_derivative_element_wise_function(f):
    f_second = jax.vmap(jax.grad(jax.grad(f)))
    return(f_second)

def check_super_sub_gaussian(x): 
    """Check if the single signal source verify the sub or supergaussian identity 
    E(xg(x)-g'(x)) > 0 
    where g : derivative of sub or super gaussian log probabilities

    Args:
        x (array): single source in T points
    """
    g_minus = derivative_element_wise_function(get_subgaussian_log_prob)
    g_minus_derivative = second_derivative_element_wise_function(get_subgaussian_log_prob)
    g_plus = derivative_element_wise_function(get_supergaussian_log_prob)
    g_plus_derivative = second_derivative_element_wise_function(get_supergaussian_log_prob)
    y_minus= x*g_minus(x) -   g_minus_derivative(x)
    y_plus = x*g_plus(x) - g_plus_derivative(x)

    if bool(jnp.mean(y_minus)>0.):
        return(g_minus)
    elif bool(jnp.mean(y_plus)>0.):
        return(g_plus)
    else: 
        return(None)
    


def log_likelihood_signal(X,w_i ): 
    """_summary_

    Args:
        x (_type_): single signal x
        w (_type_): line of W 
    """
    grad_log_proba= check_super_sub_gaussian(jnp.dot(w_i,X))
    return(grad_log_proba)



def log_likelihood_signals(X, W):
    """
    Calcul de la log-vraisemblance des signaux X pour différentes parties de la matrice de mélange W.

    Args:
    - X: Matrice de signaux de taille (n, T) où n est le nombre de signaux et T est la longueur du signal.
    - W: Matrice de mélange de taille (n, n).

    Returns:
    - log_likelihoods: Liste de log-vraisemblances pour différentes parties de W.
    """
    n, T = X.shape
    log_likelihoods = []

    for i in range(0, n):
        W_i = W[i, :]  
        log_likelihood= log_likelihood_signal(X, W_i)
        log_likelihoods.append(log_likelihood(jnp.dot(W_i,X)))

    return log_likelihoods


def update_W(X,W, mu):
    l = jnp.array(log_likelihood_signals(X,W))
    delta_W = jnp.eye(W.shape[0]) + jnp.dot(l, jnp.dot(W,X).T)
    delta_W = delta_W.dot(W)
    W_new = W + mu*delta_W
    return(W_new , l)


def make_matrix_orthonormal(W):
    """
    Rend la matrice W orthonormée en utilisant la décomposition en valeurs singulières (SVD).

    Args:
    - W: Matrice de mélange de taille (n, n).

    Returns:
    - W_orthonormal: Matrice orthonormée de même taille que W.
    """
    # Calcul de la décomposition en valeurs singulières
    U, _, _ = jnp.linalg.svd(W)

    # Utiliser la première partie de la décomposition comme matrice orthonormée
    W_orthonormal = U

    return W_orthonormal

def initialize_random_W(n_sources):
    return jax.random.normal(jax.random.PRNGKey(0), (n_sources, n_sources))

def mle_ICA(X,max_iter,eps, learning_rate ):
    W = initialize_random_W(3)
    print(W.shape)
    for iteration in range(max_iter):
        W_old = W
        W , loglikelihood_val = update_W(X=X, W=W, mu=learning_rate)
        W = make_matrix_orthonormal(W)
        new_loglikelihood = jnp.array(log_likelihood_signals(X,W))
        if jnp.linalg.norm(W-W_old) < eps:
            break
    return(W)


def update_W2(X,W, mu):
    l = jnp.array(log_likelihood_signals(X,W))
    delta_W = W+ jnp.dot(l, X.T)
    W_new = W + mu*delta_W
    return(W_new , l)

def mle_ICA2(X,max_iter,eps, learning_rate ):
    W = initialize_random_W(3)
    print(W.shape)
    for iteration in range(max_iter):
        W_old = W
        W , loglikelihood_val = update_W2(X=X, W=W, mu=learning_rate)
        W = make_matrix_orthonormal(W)
        new_loglikelihood = jnp.array(log_likelihood_signals(X,W))
        if jnp.linalg.norm(W-W_old) < eps:
            break
    return(W)