import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist


@jax.jit
def log_nu_approx(t):
    return -jnp.log(jnp.cos(jnp.pi / 2 * jnp.sqrt(t)))


@jax.jit
def nu_approx(t):
    return jnp.exp(log_nu_approx(t))


def model(x, y, dx, dy, nu=None):
    # Prior on heavy-tailedness
    if nu is None:
        # nu = numpyro.sample("nu", dist.InverseGamma(3, 10))
        t = numpyro.sample("t", dist.Uniform(0, 1))
        nu = numpyro.deterministic("nu", nu_approx(t))

    # Latent distribution of true x values
    x_true = numpyro.sample("x", dist.Normal(), sample_shape=x.shape)

    # Priors on regression parameters
    alpha = numpyro.sample("alpha", dist.Normal(0, 3))
    beta = numpyro.sample(
        "beta", dist.Normal(0, 3), sample_shape=(x.shape[0],)
    )
    sigma_int = numpyro.sample("sigma_int", dist.Cauchy(0, 5))
    sigma = numpyro.deterministic("sigma", sigma_int)

    # Linear regression model
    eps = numpyro.sample(
        "eps", dist.StudentT(nu, 0, sigma), sample_shape=y.shape
    )
    y_true = numpyro.deterministic("y", alpha + jnp.dot(beta, x_true) + eps)

    # Measure latent x and y values with error
    numpyro.sample("x_obs", dist.MultivariateStudentT(nu, x_true, dx), obs=x)
    numpyro.sample("y_obs", dist.StudentT(nu, y_true, dy), obs=y)


def tcup(data, seed=None, prior_samples=1000, **sampler_kwargs):
    # Setup random key
    if seed is None:
        seed = np.random.randint(4294967296)
    rng_key = jax.random.PRNGKey(seed)

    # Set up NUTS kernel
    kernel = numpyro.infer.NUTS(model)

    # Set up sampler
    sampler_kwargs.setdefault("num_warmup", 1000)
    sampler_kwargs.setdefault("num_samples", 1000)
    sampler_kwargs.setdefault("num_chains", 4)
    sampler_kwargs.setdefault("chain_method", "parallel")
    mcmc = numpyro.infer.MCMC(kernel, **sampler_kwargs)

    # Sample model
    rng_key, rng_key_ = jax.random.split(rng_key)
    mcmc.run(rng_key_, **data, extra_fields=["num_steps", "energy"])
    samples = mcmc.get_samples()

    # Sample posterior predictive
    rng_key, rng_key_ = jax.random.split(rng_key)
    post_pred = numpyro.infer.Predictive(model, samples)(rng_key_, **data)

    # Sample prior
    rng_key, rng_key_ = jax.random.split(rng_key)
    prior = numpyro.infer.Predictive(model, num_samples=prior_samples)(
        rng_key_, **data
    )

    # Combine results into ArviZ InferenceData object
    results = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=post_pred,
    )

    return results
