"""Bayesian k-NN Fisher-Rao energy: q replaced by the mean over B Dirichlet(dirichlet_alpha) bootstrap draws.

Temperature handling (beta_t = sigma_t^2, beta_soft floor) is shared with basic_fr.
"""
import torch
import torch.nn.functional as F

from . import basic_fr
from .driver import apply_projector

DEFAULT_B = 256
DEFAULT_DIRICHLET_ALPHA = 1.0


def soft_posterior_bb(y, neighbors, beta_t, B=None, dirichlet_alpha=None, return_var=False, beta_soft=None,
                       generator=None, P=None):
    """Bootstrap-averaged posterior over the k neighbors.

    P: optional tangent projector, applied to g before the posterior (as in basic_fr.fisher_rao_energy).
    beta_soft: optional per-sample temperature floor [B]; needed at high D, where the bootstrap alone stays one-hot.
    generator: torch.Generator, or a list with one per row of y (batch-independent draws); None uses the global RNG.
    Returns (q_mean [B, k], g [B, k, D]), plus the per-atom variance over draws if return_var."""
    B = B or DEFAULT_B
    dirichlet_alpha = dirichlet_alpha or DEFAULT_DIRICHLET_ALPHA
    k = neighbors.shape[1]
    beta_col = beta_t.view(-1, 1)
    # reshape(-1, 1): a [B] beta_soft against [B, 1] would broadcast to [B, B]
    beta_soft_col = (beta_col if beta_soft is None
                      else torch.maximum(beta_col, beta_soft.reshape(-1, 1).to(beta_col)))

    g = y.unsqueeze(1) - neighbors
    if P is not None:
        g = apply_projector(P, g)
    E = 0.5 * (g ** 2).sum(-1)

    concentration = torch.full((k,), dirichlet_alpha, device=y.device, dtype=y.dtype)
    if generator is None:
        p = torch.distributions.Dirichlet(concentration).sample((B,))
    elif isinstance(generator, (list, tuple)):
        gamma = torch.stack([torch._standard_gamma(concentration.expand(B, k).to(gen.device), generator=gen)
                             for gen in generator]).to(y.device)
        p = gamma / gamma.sum(-1, keepdim=True)          # [Bq, B, k]
    else:
        # Dirichlet.sample takes no generator, so normalize Gamma draws made on the generator's device
        gamma = torch._standard_gamma(concentration.expand(B, k).to(generator.device),
                                      generator=generator).to(y.device)
        p = gamma / gamma.sum(-1, keepdim=True)
    logp = torch.log(p.clamp_min(1e-12))
    if logp.dim() == 2:
        logp = logp.unsqueeze(0)
    logits = logp - E.unsqueeze(1) / beta_soft_col.unsqueeze(-1)  # [Bq, B, k]
    q_b = F.softmax(logits, dim=-1)
    q_mean = q_b.mean(dim=1)

    if return_var:
        q_var = q_b.var(dim=1)
        return q_mean, g, q_var
    return q_mean, g


def fisher_rao_energy_bb(y, neighbors, beta_t, B=None, dirichlet_alpha=None, P=None, beta_soft=None,
                          generator=None):
    """basic_fr.fisher_rao_energy with q replaced by soft_posterior_bb's q_mean. Returns (I, q_mean, g)."""
    q, g = soft_posterior_bb(y, neighbors, beta_t, B=B, dirichlet_alpha=dirichlet_alpha, beta_soft=beta_soft,
                              generator=generator, P=P)
    Eq_g = (q.unsqueeze(-1) * g).sum(1)
    Eq_g2 = (q * (g ** 2).sum(-1)).sum(1)
    I = (Eq_g2 - (Eq_g ** 2).sum(-1)) / beta_t ** 2
    return I, q, g


def make_knn_bayesian_fr_energy_fn(index, k, beta_t_fn, B=None, dirichlet_alpha=None, q_target=None,
                                    generator=None):
    """energy_fn(y) -> I [B]; q_target sets the plug-in q_max, so the achieved q_mean max is somewhat lower."""
    def energy_fn(y):
        with torch.no_grad():
            neighbors, sq_d = basic_fr.ann_query(index, y, k)
        I, _, _ = fisher_rao_energy_bb(y, neighbors, beta_t_fn(), B=B, dirichlet_alpha=dirichlet_alpha,
                                        beta_soft=basic_fr.beta_soft_for(sq_d, q_target),
                                        generator=generator)
        return I
    return energy_fn


def make_knn_bayesian_projected_energy_fn(index, k, beta_t_fn, B=None, dirichlet_alpha=None, q_target=None,
                                           generator=None):
    """energy_fn(y, P) -> I [B]; tangent-projected make_knn_bayesian_fr_energy_fn."""
    def energy_fn(y, P):
        with torch.no_grad():
            neighbors, sq_d = basic_fr.ann_query(index, y, k)
        I, _, _ = fisher_rao_energy_bb(y, neighbors, beta_t_fn(), B=B, dirichlet_alpha=dirichlet_alpha, P=P,
                                        beta_soft=basic_fr.beta_soft_for(sq_d, q_target),
                                        generator=generator)
        return I
    return energy_fn
