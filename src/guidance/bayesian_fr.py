"""Energy function for the Bayesian variant (Dirichlet-bootstrap posterior
over basic_fr's k-NN Fisher-Rao energy) -- for use with driver.guided_reverse_loop.

Same k-NN retrieval as basic_fr (via ann_query), but the softmax posterior q
over the k retrieved atoms is replaced by q_bar: B draws of a
Dirichlet(dirichlet_alpha) prior over those k atoms, each combined with the
data term into a Gibbs posterior, then averaged. q_bar -> q (uniform 1/k
prior) only as dirichlet_alpha -> infinity; dirichlet_alpha=1.0 (flat
Dirichlet) is the canonical bootstrap.

beta_t is pinned to sigma_t^2 exactly as in basic_fr -- the bootstrap
replaces the prior over atoms, not the temperature. The softmax temperature
itself is overridden per sample via beta_soft, exactly as in basic_fr, so
q_target-based tempering is shared across all three energies (see
basic_fr.beta_soft_for). Because beta_soft depends only on the retrieved
squared distances, basic_fr / bayesian_fr / bayesian_fr_v2 all receive an
IDENTICAL temperature at the same y: the three differ only in how E/beta is
turned into a posterior, which is what makes them directly comparable.
"""
import torch
import torch.nn.functional as F

from . import basic_fr
from .driver import apply_projector

DEFAULT_B = 256
DEFAULT_DIRICHLET_ALPHA = 1.0


def soft_posterior_bb(y, neighbors, beta_t, B=None, dirichlet_alpha=None, return_var=False, beta_soft=None,
                       generator=None):
    """y: [B, D], neighbors: [B, k, D] (the k retrieved training points, same
    as basic_fr.fisher_rao_energy). beta_t: [B], pinned to sigma_t^2.

    beta_soft: optional per-sample softmax temperature [B] (typically
    basic_fr.beta_soft_for(sq_dists, q_target)). The softmax uses
    max(beta_t, beta_soft) elementwise; the caller (fisher_rao_energy_bb)
    still divides by the TRUE, unfloored beta_t^2 for the outer scale. None
    reproduces the untempered behavior.

    This variant NEEDS the tempering as much as basic_fr does -- the
    Dirichlet bootstrap does NOT protect it. Measured on a 1000-image CIFAR
    subset at progress 0.90 (beta_t = 0.0867, k = 15), untempered: q_max =
    1.000000 and ||grad I|| EXACTLY 0.0 for 64/64 samples, at both
    dirichlet_alpha = 0.5 and 1.0. The reason is that at D = 3072 the energy
    gaps between neighbors (~40-500) dwarf log p (~O(1-10)), so every one of
    the B draws collapses onto the SAME atom, and the average of B identical
    one-hots is still one-hot. The bootstrap only spreads q_mean when
    beta * log p is comparable to the energy gap, which is a low-dimension
    regime. Contrast bayesian_fr_v2, whose alpha_0/kappa shrinkage bounds
    q_bar_max below 1 structurally and so cannot hit this dead zone.

    generator: optional torch.Generator for the Dirichlet bootstrap draw --
    without it, the draw consumes the global RNG, so results vary run to run
    even with a fixed z_init. Pass the same generator used to seed sampling
    for reproducibility.

    Returns (q_mean [B, k], g [B, k, D]) -- g is the same neighbor-difference
    tensor basic_fr.fisher_rao_energy computes, so fisher_rao_energy_bb can
    reuse it directly. Pass return_var=True to also get q_b's per-atom
    variance across the B draws (epistemic spread), returned as a third value.
    """
    B = B or DEFAULT_B
    dirichlet_alpha = dirichlet_alpha or DEFAULT_DIRICHLET_ALPHA
    k = neighbors.shape[1]
    beta_col = beta_t.view(-1, 1)
    # reshape(-1, 1) is load-bearing: beta_col is [B, 1] and beta_soft is [B], so a bare
    # clamp(min=beta_soft) would broadcast to [B, B] SILENTLY rather than elementwise.
    beta_soft_col = (beta_col if beta_soft is None
                      else torch.maximum(beta_col, beta_soft.reshape(-1, 1).to(beta_col)))

    g = y.unsqueeze(1) - neighbors                      # [Bq, k, D]
    E = 0.5 * (g ** 2).sum(-1)                            # [Bq, k]

    concentration = torch.full((k,), dirichlet_alpha, device=y.device, dtype=y.dtype)
    if generator is None:
        p = torch.distributions.Dirichlet(concentration).sample((B,))  # [B, k]
    else:
        # torch.distributions.Dirichlet.sample has no generator= hook, so draw
        # the underlying Gamma variates directly (same reparameterization it
        # uses internally) against the given generator, then normalize.
        gamma = torch._standard_gamma(concentration.expand(B, k), generator=generator)
        p = gamma / gamma.sum(-1, keepdim=True)          # [B, k]
    logp = torch.log(p.clamp_min(1e-12))                  # [B, k]
    logits = logp.unsqueeze(0) - E.unsqueeze(1) / beta_soft_col.unsqueeze(-1)  # [Bq, B, k]
    q_b = F.softmax(logits, dim=-1)                       # [Bq, B, k]
    q_mean = q_b.mean(dim=1)                               # [Bq, k]

    if return_var:
        q_var = q_b.var(dim=1)                             # [Bq, k]
        return q_mean, g, q_var
    return q_mean, g


def fisher_rao_energy_bb(y, neighbors, beta_t, B=None, dirichlet_alpha=None, P=None, beta_soft=None,
                          generator=None):
    """Bayesian counterpart of basic_fr.fisher_rao_energy: identical
    variance-of-g formula, with q replaced by the bootstrap-averaged q_bar
    from soft_posterior_bb. P: optional tangent projector, applied to g
    before anything else, same contract as basic_fr.fisher_rao_energy.
    beta_soft: see soft_posterior_bb -- only affects the posterior's
    temperature, the outer 1/beta_t^2 below always uses the true beta_t.
    generator: see soft_posterior_bb.
    Returns (I [B], q_mean [B, k], g [B, k, D])."""
    g = y.unsqueeze(1) - neighbors
    if P is not None:
        g = apply_projector(P, g)
    q, _ = soft_posterior_bb(y, neighbors, beta_t, B=B, dirichlet_alpha=dirichlet_alpha, beta_soft=beta_soft,
                              generator=generator)
    Eq_g = (q.unsqueeze(-1) * g).sum(1)
    Eq_g2 = (q * (g ** 2).sum(-1)).sum(1)
    I = (Eq_g2 - (Eq_g ** 2).sum(-1)) / beta_t ** 2
    return I, q, g


def make_knn_bayesian_fr_energy_fn(index, k, beta_t_fn, B=None, dirichlet_alpha=None, q_target=None,
                                    generator=None):
    """Returns energy_fn(y) -> I [B] for driver.guidance_grad /
    guided_reverse_loop (the AMBIENT, unprojected energy) -- Bayesian
    counterpart of basic_fr.make_knn_fr_energy_fn. q_target: setpoint for the
    PLUG-IN q_max, resolved through basic_fr.beta_soft_for, i.e. the same
    temperature basic_fr would use on the same neighbors -- so a sweep over
    q_target is directly comparable across the three energies. Note the
    resulting q_mean's max lands somewhat BELOW the setpoint here, since the
    Dirichlet average spreads mass further than the plain softmax; log the
    achieved q_mean.max() rather than assuming it equals q_target.
    generator: see soft_posterior_bb -- pass the same generator seeding the
    sampler's z_init for reproducible runs."""
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
    """Returns energy_fn(y, P) -> I [B] for
    driver.projected_guidance_grad/make_projected_guidance_fn -- the tangent-
    restricted counterpart of make_knn_bayesian_fr_energy_fn. q_target: see
    make_knn_bayesian_fr_energy_fn. generator: see soft_posterior_bb."""
    def energy_fn(y, P):
        with torch.no_grad():
            neighbors, sq_d = basic_fr.ann_query(index, y, k)
        I, _, _ = fisher_rao_energy_bb(y, neighbors, beta_t_fn(), B=B, dirichlet_alpha=dirichlet_alpha, P=P,
                                        beta_soft=basic_fr.beta_soft_for(sq_d, q_target),
                                        generator=generator)
        return I
    return energy_fn
