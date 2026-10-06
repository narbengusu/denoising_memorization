"""AMG dissimilarity guidance (Chen, Liu and Xu, CVPR 2024, arXiv:2404.00922) for unconditional models.

- Similarity (Eq. 8): sigma_t = -l2(x0_hat, n_0) / (alpha * mean l2 to the k nearest), k = 50, alpha = 0.5; neighbours
  searched without gradient. For latent models x0 is the decoded image (Supp. D).
- Indicator (Eq. 10): guide while sigma_t > lambda_t; lambda_t = a + (b - a) exp(-c t) (Supp. Eq. 19), a = -1.95,
  b = -1.5, c = 0.025 (the paper prints -0.025, inconsistent with its text), t = 250 * noise_level.
- Updates: eps += c3 sqrt(1 - abar) grad (Eqs. 10, 11, 17); EDM: D -= c3 s^2 grad; DDPM mean: mu -= c Sigma grad
  (Supp. Eqs. 21-22). The paper gives no c3 or c, so they are swept.

Usage inside a sampler step (x_in = x.detach().requires_grad_(True)):

    with torch.enable_grad():
        eps = model(x_in, t)
        x0 = predict_x0(x_in, eps)
        shift, nl2, on = eps_guidance(x_in, x0, train, noise_level, c3, sqrt_one_minus_abar)
    eps = eps.detach() + shift
"""
import numpy as np
import torch

ALPHA, NEIGHBORS = 0.5, 50
SCHEDULE_A, SCHEDULE_B, SCHEDULE_C, SCHEDULE_STEPS = -1.95, -1.5, 0.025, 250


def schedule(noise_level):
    """lambda_t (Supp. Eq. 19): -1.95 at pure noise, -1.5 at the last step."""
    return SCHEDULE_A + (SCHEDULE_B - SCHEDULE_A) * np.exp(-SCHEDULE_C * SCHEDULE_STEPS * noise_level)


def l2(a, b):
    """Pixel RMS distance."""
    return ((a - b) ** 2).mean(-1).sqrt()


def nearest(y, train, k, chunk=100):
    """Exact-l2 k nearest training rows of each row of y: (values, indices), ascending; cdist candidates rescored."""
    values, indices = [], []
    for s in range(0, len(y), chunk):
        yc = y[s:s + chunk]
        cand = torch.cdist(yc, train).topk(min(4 * k, len(train)), largest=False).indices
        top = l2(yc[:, None], train[cand]).topk(k, largest=False)
        values.append(top.values); indices.append(cand.gather(1, top.indices))
    return torch.cat(values), torch.cat(indices)


def nl2(y, train, alpha=ALPHA, neighbors=NEIGHBORS):
    """nL2 (Eq. 8, without the sign) of each sample of y against the training set; differentiable in y."""
    flat, train_flat = y.reshape(len(y), -1), train.reshape(len(train), -1)
    k = min(neighbors, len(train_flat))
    with torch.no_grad():
        _, indices = nearest(flat.detach(), train_flat, k)
    distances = l2(flat[:, None], train_flat[indices])
    return distances[:, 0] / (alpha * distances.mean(1))


def _similarity_gradient(x_in, x0, train, noise_level, c):
    """(grad_{x_in} sigma_t masked to c > 0 and sigma_t > lambda_t, nL2, mask)."""
    sigma = -nl2(x0, train)
    on = (sigma.detach() > float(schedule(noise_level))) & (c > 0)
    grad = torch.autograd.grad((sigma * on).sum(), x_in)[0] if on.any() else torch.zeros_like(x_in)
    return grad, -sigma.detach(), on


def _per_sample(v, like):
    v = torch.as_tensor(v, dtype=like.dtype, device=like.device)
    return v.reshape(-1, *([1] * (like.ndim - 1))) if v.ndim else v


def _strength(c, like):
    return torch.as_tensor(c, dtype=like.dtype, device=like.device).expand(len(like))


def eps_guidance(x_in, x0, train, noise_level, c3, sqrt_one_minus_abar):
    """Eqs. 10, 17: (shift, nl2, on) with eps_guided = eps + shift; c3 scalar or [B]."""
    c3 = _strength(c3, x_in)
    grad, value, on = _similarity_gradient(x_in, x0, train, noise_level, c3)
    return _per_sample(c3, x_in) * _per_sample(sqrt_one_minus_abar, x_in) * grad, value, on


def denoiser_guidance(x_in, x0, train, noise_level, c3, sigma):
    """Eqs. 10, 17 for an EDM denoiser: returns (shift, nl2, on) with D_guided = D + shift."""
    c3 = _strength(c3, x_in)
    grad, value, on = _similarity_gradient(x_in, x0, train, noise_level, c3)
    return -_per_sample(c3, x_in) * _per_sample(sigma, x_in) ** 2 * grad, value, on


def mean_guidance(x_in, x0, train, noise_level, c, variance):
    """Supp. Eqs. 21-22: (shift, nl2, on) with mu_guided = mu + shift; variance is the model's Sigma."""
    c = _strength(c, x_in)
    grad, value, on = _similarity_gradient(x_in, x0, train, noise_level, c)
    return -_per_sample(c, x_in) * variance * grad, value, on
