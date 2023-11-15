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
    return jnp.sum(-2*(jnp.logaddexp(x,-x))-math.log(2))

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
    return jnp.sum(jnp.logaddexp(x, -x) - math.log(math.sqrt(2*math.pi)) - 1/2)

def derivative_log_supergaussian(x): 
    return(jnp.grad(get_supergaussian_log_prob(x)))

def derivative_log_subgaussian(x): 
    return(jnp.grad(get_subgaussian_log_prob(x)))
                   
def check_super_sub_gaussian(x): 
    """Check if the single signal source verify the sub or supergaussian identity 
    E(xg(x)-g'(x)) > 0 
    where g : derivative of sub or super gaussian log probabilities

    Args:
        x (array): single source in T points
    """