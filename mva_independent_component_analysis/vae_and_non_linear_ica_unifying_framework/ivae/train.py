import optax
from mva_independent_component_analysis.utils.math import logaddexp
from mva_independent_component_analysis.utils.metrics import mean_corr_coef
from .nets import IVAE
from optax import adam
from optax import piecewise_constant_schedule
from mva_independent_component_analysis.vae_and_non_linear_ica_unifying_framework.data import DataSet
import jax.numpy as jnp
import jax
from .exponential_family import logdensity_normal
import flax


def create_batch(OP_key, dataset, batch_size):
    n = dataset.len // batch_size

    def split_and_shuffle(key, x):
        x = jnp.asarray(jnp.split(x, n))
        x = jax.random.permutation(key, x)
        return x

    return split_and_shuffle(OP_key, dataset.x), split_and_shuffle(OP_key, dataset.s), split_and_shuffle(OP_key,
                                                                                                         dataset.u)


def train_and_evaluate(OP_key, dataset, model_cfg, learning_cfg):
    assert isinstance(dataset, DataSet)

    data_dim, latent_dim, aux_dim = dataset.get_dims()
    model_cfg.update({'data_dim': data_dim, 'latent_dim': latent_dim, 'aux_dim': aux_dim})

    # learning parameters
    lr = learning_cfg.pop('lr', 1e-2)
    batch_size = learning_cfg.pop('batch_size', 64)
    epochs = learning_cfg.pop('epochs', 100)
    # fixed Gaussian noise: \varepsilon ~ N(0, sigma^2Id)
    var_decoder = 0.1

    N = dataset.len
    assert N % batch_size == 0
    n = N // batch_size
    key1, key = jax.random.split(OP_key, 2)
    batches = create_batch(key1, dataset, batch_size)

    # order: x, s, u
    x_shape, u_shape = batches[0][0].shape, batches[2][0].shape

    model = IVAE(**model_cfg)
    variables = model.init(key, key=key1, x=jnp.empty(x_shape), u=jnp.empty(u_shape))
    state, params = flax.core.pop(variables, 'params')
    del variables
    scheduler = piecewise_constant_schedule(lr, {int(0.8 * epochs): 0.8})
    optimizer = adam(scheduler)
    opt_state = optimizer.init(params)
    batch_keys = jax.random.split(key, n)

    def train_step(state, opt_state, params, z_rng_batch, x_batch, u_batch, a, b, c, d):
        """
        See: https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/state_params.html
             https://github.com/ilkhem/iVAE
             Assuming z \mid u Gaussian-prior.
        """

        def batch_loss(params):
            def loss_fn(z_rng, x, u):
                f, g, v, z, l = model.apply(
                    {'params': params}, z_rng, x, u  # **state
                )
                M, d_latent = z.shape
                logpx = jnp.sum(logdensity_normal(x, f, var_decoder), axis=-1)
                logqs_cux = jnp.sum(logdensity_normal(z, g, v), axis=-1)
                logps_cu = jnp.sum(logdensity_normal(z, 0., l), axis=-1)

                logqs_tmp = logdensity_normal(z.reshape(M, 1, d_latent), g.reshape(1, M, d_latent),
                                              v.reshape(1, M, d_latent))
                logqs = logaddexp(jnp.sum(logqs_tmp, axis=-1), axis=1) - jnp.log(M * N)
                logqs_i = jnp.sum(logaddexp(logqs_tmp, axis=1) - jnp.log(M * N), axis=-1)

                elbo = -jnp.mean(
                    (a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps_cu)))
                return elbo, z

            loss, updated_state = jax.vmap(loss_fn, out_axes=(0, 0))(z_rng_batch, x_batch,
                                                                     u_batch)  # out_axes = (0, None)?
            return loss.mean(), updated_state

        (loss, updated_state), grads = jax.value_and_grad(
            batch_loss, has_aux=True,
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return opt_state, params, updated_state, loss
    
    mcc_scores=[]
    for epoch in range(epochs):
        opt_state, params, state, loss = train_step(state, opt_state, params, batch_keys, batches[0], batches[2],
                                                    **learning_cfg)
        mean_corr_coeffs, _ = jax.vmap(mean_corr_coef, in_axes=(0, 0))(state, batches[1])
        mean_corr_coeff = jnp.mean(mean_corr_coeffs)
        print(f"Epoch: {epoch}; Loss: {loss}; Mean correlation coefficient: {mean_corr_coeff}")
        mcc_scores.append(mean_corr_coeff)
    return mcc_scores
