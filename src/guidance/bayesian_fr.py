"""Energy function for the Bayesian variant (Dirichlet-bootstrap posterior
over basic_fr's k-NN Fisher-Rao energy) -- for use with driver.guided_reverse_loop.

Same k-NN retrieval as basic_fr (via ann_query), but the softmax posterior q
over the k retrieved atoms is replaced by q_bar: B draws of a
Dirichlet(dirichlet_alpha) prior over those k atoms, each combined with the
data term into a Gibbs posterior, then averaged. q_bar -> q (uniform 1/k
prior) only as dirichlet_alpha -> infinity; dirichlet_alpha=1.0 (flat
Dirichlet) is the canonical bootstrap.

beta_t is pinned to sigma_t^2 exactly as in basic_fr -- the bootstrap
replaces the prior over atoms, not the temperature.
"""
import torch
import torch.nn.functional as F

from . import basic_fr
from .driver import apply_projector

DEFAULT_B = 256
DEFAULT_DIRICHLET_ALPHA = 1.0


def soft_posterior_bb(y, neighbors, beta_t, B=None, dirichlet_alpha=None, return_var=False, beta_floor=None):
    """y: [B, D], neighbors: [B, k, D] (the k retrieved training points, same
    as basic_fr.fisher_rao_energy). beta_t: [B], pinned to sigma_t^2.

    beta_floor: see basic_fr.fisher_rao_energy's docstring -- this Gibbs
    posterior has the identical vanishing-temperature pathology (beta_t -> 0
    late in sampling makes q_b saturate to a near-one-hot per draw, same as
    basic_fr's plain softmax, just Dirichlet-perturbed), so it needs the same
    decoupling fix: when given, the softmax uses max(beta_t, beta_floor) as
    its temperature; the caller (fisher_rao_energy_bb) still divides by the
    TRUE, unfloored beta_t^2 for the outer scale. None reproduces the
    original, undecoupled behavior.

    Returns (q_mean [B, k], g [B, k, D]) -- g is the same neighbor-difference
    tensor basic_fr.fisher_rao_energy computes, so fisher_rao_energy_bb can
    reuse it directly. Pass return_var=True to also get q_b's per-atom
    variance across the B draws (epistemic spread), returned as a third value.
    """
    B = B or DEFAULT_B
    dirichlet_alpha = dirichlet_alpha or DEFAULT_DIRICHLET_ALPHA
    k = neighbors.shape[1]
    beta_col = beta_t.view(-1, 1)
    beta_soft_col = beta_col if beta_floor is None else beta_col.clamp(min=beta_floor)

    g = y.unsqueeze(1) - neighbors                      # [Bq, k, D]
    E = 0.5 * (g ** 2).sum(-1)                            # [Bq, k]

    dirichlet = torch.distributions.Dirichlet(
        torch.full((k,), dirichlet_alpha, device=y.device, dtype=y.dtype))
    p = dirichlet.sample((B,))                            # [B, k]
    logp = torch.log(p.clamp_min(1e-12))                  # [B, k]
    logits = logp.unsqueeze(0) - E.unsqueeze(1) / beta_soft_col.unsqueeze(-1)  # [Bq, B, k]
    q_b = F.softmax(logits, dim=-1)                       # [Bq, B, k]
    q_mean = q_b.mean(dim=1)                               # [Bq, k]

    if return_var:
        q_var = q_b.var(dim=1)                             # [Bq, k]
        return q_mean, g, q_var
    return q_mean, g


def fisher_rao_energy_bb(y, neighbors, beta_t, B=None, dirichlet_alpha=None, P=None, beta_floor=None):
    """Bayesian counterpart of basic_fr.fisher_rao_energy: identical
    variance-of-g formula, with q replaced by the bootstrap-averaged q_bar
    from soft_posterior_bb. P: optional tangent projector, applied to g
    before anything else, same contract as basic_fr.fisher_rao_energy.
    beta_floor: see soft_posterior_bb -- only affects the posterior's
    temperature, the outer 1/beta_t^2 below always uses the true beta_t.
    Returns (I [B], q_mean [B, k], g [B, k, D])."""
    g = y.unsqueeze(1) - neighbors
    if P is not None:
        g = apply_projector(P, g)
    q, _ = soft_posterior_bb(y, neighbors, beta_t, B=B, dirichlet_alpha=dirichlet_alpha, beta_floor=beta_floor)
    Eq_g = (q.unsqueeze(-1) * g).sum(1)
    Eq_g2 = (q * (g ** 2).sum(-1)).sum(1)
    I = (Eq_g2 - (Eq_g ** 2).sum(-1)) / beta_t ** 2
    return I, q, g


def make_knn_bayesian_fr_energy_fn(index, k, beta_t_fn, B=None, dirichlet_alpha=None, beta_floor=None):
    """Returns energy_fn(y) -> I [B] for driver.guidance_grad /
    guided_reverse_loop (the AMBIENT, unprojected energy) -- Bayesian
    counterpart of basic_fr.make_knn_fr_energy_fn. beta_floor: see
    fisher_rao_energy_bb."""
    def energy_fn(y):
        with torch.no_grad():
            neighbors, _ = basic_fr.ann_query(index, y, k)
        I, _, _ = fisher_rao_energy_bb(y, neighbors, beta_t_fn(), B=B, dirichlet_alpha=dirichlet_alpha,
                                        beta_floor=beta_floor)
        return I
    return energy_fn


def make_knn_bayesian_projected_energy_fn(index, k, beta_t_fn, B=None, dirichlet_alpha=None, beta_floor=None):
    """Returns energy_fn(y, P) -> I [B] for
    driver.projected_guidance_grad/make_projected_guidance_fn -- the tangent-
    restricted counterpart of make_knn_bayesian_fr_energy_fn. beta_floor: see
    fisher_rao_energy_bb."""
    def energy_fn(y, P):
        with torch.no_grad():
            neighbors, _ = basic_fr.ann_query(index, y, k)
        I, _, _ = fisher_rao_energy_bb(y, neighbors, beta_t_fn(), B=B, dirichlet_alpha=dirichlet_alpha, P=P,
                                        beta_floor=beta_floor)
        return I
    return energy_fn
