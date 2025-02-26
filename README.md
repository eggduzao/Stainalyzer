# Stainalyzer

- Open and publicly available tool to automatically evaluate immunohistochemically-stained antibodies

---

# Latent Stochastic Process

We define the generative process as a **stochastic differential equation (SDE)**:

The stochastic differential equation governing the process is:

$$
d\mathbf{z}_t = \mu(\mathbf{z}_t, t) dt + \sigma(\mathbf{z}_t, t) d\mathbf{W}_t,
$$

where $\mathbf{z}_t$ is the latent representation at time $t$, $\mu(\mathbf{z}_t, t)$ is the drift term, $\sigma(\mathbf{z}_t, t)$ is the diffusion coefficient, and $\mathbf{W}_t$ is a Wiener process modeling Brownian motion.

The transition density of the process is governed by the Fokker-Planck equation:

$$
\frac{\partial p(\mathbf{z}, t)}{\partial t} = - \nabla \cdot (\mu(\mathbf{z}, t) p(\mathbf{z}, t)) + \frac{1}{2} \nabla^2 (\sigma^2(\mathbf{z}, t) p(\mathbf{z}, t)).
$$

# Variational Inference

The encoder network approximates the posterior \( q_\phi(\mathbf{z} | \mathbf{x}) \) via the **reparameterization trick**:

$$
\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I).
$$

The evidence lower bound (ELBO) is:

$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} \left[ \log p_\theta(\mathbf{x} | \mathbf{z}) \right] - D_{KL} \left( q_\phi(\mathbf{z} | \mathbf{x}) || p(\mathbf{z}) \right).
$$

# Diffusion Model Integration

The forward process gradually adds Gaussian noise to the latent space:

$$
q(\mathbf{z}_t | \mathbf{z}_0) = \mathcal{N}(\mathbf{z}_t; \alpha_t \mathbf{z}_0, \sigma_t^2 I),
$$

where \( \alpha_t \) and \( \sigma_t \) define the noise schedule. The reverse process learns to denoise step-by-step:

$$
p_\theta(\mathbf{z}_{t-1} | \mathbf{z}_t) = \mathcal{N}(\mathbf{z}_{t-1}; \mu_\theta(\mathbf{z}_t, t), \Sigma_\theta(\mathbf{z}_t, t)).
$$

# Training and Optimization

We minimize the variational loss function:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{VAE}} \mathcal{L}(\theta, \phi) + \lambda_{\text{diff}} \mathcal{L}_{\text{diffusion}},
$$

where the diffusion loss is given by:

$$
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, \mathbf{z}_t} \left[ || \mathbf{z}_t - \mu_\theta(\mathbf{z}_t, t) ||^2 \right].
$$


