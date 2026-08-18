"""Energy function for the closed-form Bayesian variant (bayesianv2): a
Dirichlet(alpha_0 * 1_k) prior placed directly on the unknown local
assignment pi over the k retrieved atoms, updated with basic_fr's plug-in
q^beta(y) as fractional evidence with pseudo-count kappa -- for use with
driver.guided_reverse_loop.

This is the closed-form counterpart to bayesian_fr.py's Dirichlet-bootstrap
variant: instead of Monte-Carlo-averaging B draws of q_b over a Dirichlet
prior on the ATOM WEIGHTS, the posterior over pi is itself Dirichlet (general
Bayesian updating, Bissiri et al. 2016) with closed-form mean

    a_j = alpha_0 + kappa * q_j(y),   a0 = k*alpha_0 + kappa,   q_bar_j = a_j/a0

and the posterior expectation of the plug-in Fisher-Rao energy is ALSO
closed-form (no sampling):

    E[I(q)] = a0/(a0+1) * I(q_bar)

alpha_0 controls how strongly q_bar is pulled toward uniform (sparsity-
preventing / boundary-overfitting correction); kappa controls how much the
plug-in evidence q^beta(y) is trusted. Both are cheaper than bayesian_fr's
bootstrap: no Dirichlet sampling, no B-fold softmax, a single extra convex
combination on top of basic_fr's own q.

beta_t is pinned to sigma_t^2 exactly as in basic_fr/bayesian_fr -- alpha_0
and kappa reshape the ASSIGNMENT prior, not the softmax temperature.
"""
import torch

from . import basic_fr
from .driver import apply_projector

DEFAULT_ALPHA0 = 1.0
DEFAULT_KAPPA = 1.0


def posterior_mean_q_v2(y, neighbors, beta_t, alpha_0=None, kappa=None, beta_floor=None):
    """y: [B, D], neighbors: [B, k, D] (the k retrieved training points, same
    as basic_fr.fisher_rao_energy). beta_t: [B], pinned to sigma_t^2.

    beta_floor: see basic_fr.fisher_rao_energy's docstring -- q here is the
    exact same plug-in softmax as basic_fr's own q, so it has the identical
    vanishing-temperature pathology (saturates to an exact one-hot as
    beta_t -> 0 late in sampling). When given, only this softmax uses
    max(beta_t, beta_floor); q_bar's alpha_0/kappa shrinkage and a0 are
    otherwise unaffected. None reproduces the original, undecoupled behavior.

    Returns (q_bar [B, k], g [B, k, D], a0 [B]) -- q_bar is the closed-form
    Dirichlet posterior mean over pi, g is the (unprojected) neighbor-
    difference tensor basic_fr.fisher_rao_energy also computes. a0 is
    broadcast to [B] (it does not depend on y) so callers can use it directly
    in the a0/(a0+1) shrinkage factor without a separate scalar/tensor branch."""
    alpha_0 = DEFAULT_ALPHA0 if alpha_0 is None else alpha_0
    kappa = DEFAULT_KAPPA if kappa is None else kappa
    k = neighbors.shape[1]
    beta_col = beta_t.view(-1, 1)
    beta_soft_col = beta_col if beta_floor is None else beta_col.clamp(min=beta_floor)

    g = y.unsqueeze(1) - neighbors                        # [B, k, D]
    E = 0.5 * (g ** 2).sum(-1)                             # [B, k]
    q = torch.softmax(-E / beta_soft_col, dim=-1)          # [B, k], basic_fr's own plug-in q^beta(y)

    a0 = k * alpha_0 + kappa
    q_bar = (alpha_0 + kappa * q) / a0                     # [B, k]
    a0_vec = q_bar.new_full((q_bar.shape[0],), a0)
    return q_bar, g, a0_vec


def fisher_rao_energy_v2(y, neighbors, beta_t, alpha_0=None, kappa=None, P=None, beta_floor=None):
    """Closed-form Bayesian counterpart of basic_fr.fisher_rao_energy /
    bayesian_fr.fisher_rao_energy_bb: E(y) = a0/(a0+1) * I(q_bar), the exact
    posterior expectation of the plug-in Fisher-Rao energy under the
    Dirichlet posterior on pi -- no bootstrap draws (contrast
    bayesian_fr.fisher_rao_energy_bb's B-sample Monte Carlo average).

    q_bar (the assignment posterior mean) is computed from the UNPROJECTED
    neighbor distances via posterior_mean_q_v2, matching bayesian_fr.py's own
    convention (its soft_posterior_bb also ignores P); P only restricts the
    variance-of-g term g itself, same contract as basic_fr.fisher_rao_energy.
    beta_floor: see posterior_mean_q_v2 -- only affects q's own temperature;
    the outer 1/beta_t^2 below always uses the true beta_t.
    Returns (E [B], q_bar [B, k], g [B, k, D])."""
    q_bar, _, a0 = posterior_mean_q_v2(y, neighbors, beta_t, alpha_0, kappa, beta_floor=beta_floor)

    g = y.unsqueeze(1) - neighbors
    if P is not None:
        g = apply_projector(P, g)
    Eq_g = (q_bar.unsqueeze(-1) * g).sum(1)
    Eq_g2 = (q_bar * (g ** 2).sum(-1)).sum(1)
    I_qbar = (Eq_g2 - (Eq_g ** 2).sum(-1)) / beta_t ** 2
    E = (a0 / (a0 + 1)) * I_qbar
    return E, q_bar, g


def make_knn_bayesian_v2_fr_energy_fn(index, k, beta_t_fn, alpha_0=None, kappa=None, beta_floor=None):
    """Returns energy_fn(y) -> E [B] for driver.guidance_grad /
    guided_reverse_loop (the AMBIENT, unprojected energy) -- bayesianv2
    counterpart of basic_fr.make_knn_fr_energy_fn / bayesian_fr's bootstrap
    factory. beta_floor: see fisher_rao_energy_v2."""
    def energy_fn(y):
        with torch.no_grad():
            neighbors, _ = basic_fr.ann_query(index, y, k)
        E, _, _ = fisher_rao_energy_v2(y, neighbors, beta_t_fn(), alpha_0=alpha_0, kappa=kappa,
                                        beta_floor=beta_floor)
        return E
    return energy_fn


def make_knn_bayesian_v2_projected_energy_fn(index, k, beta_t_fn, alpha_0=None, kappa=None, beta_floor=None):
    """Returns energy_fn(y, P) -> E [B] for
    driver.projected_guidance_grad/make_projected_guidance_fn -- the tangent-
    restricted counterpart of make_knn_bayesian_v2_fr_energy_fn. beta_floor:
    see fisher_rao_energy_v2."""
    def energy_fn(y, P):
        with torch.no_grad():
            neighbors, _ = basic_fr.ann_query(index, y, k)
        E, _, _ = fisher_rao_energy_v2(y, neighbors, beta_t_fn(), alpha_0=alpha_0, kappa=kappa, P=P,
                                        beta_floor=beta_floor)
        return E
    return energy_fn
