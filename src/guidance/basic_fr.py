"""k-NN Fisher-Rao energy (Algorithm 1) for driver.guided_reverse_loop.

I_beta(y) = (1/beta^2) [E_q||g||^2 - ||E_q g||^2], g_i = y - Y_i, q = softmax(-E_i / beta) over the k nearest Y_i.
"""
import torch

from .driver import apply_projector


def build_flat_index(Y_flat):
    """Y_flat: [N, D]; placeholder for an ANN index."""
    return Y_flat.detach()


def ann_query(index, y, k, chunk_size=20_000):
    """Exact chunked brute-force k-NN of y [B, D]. Returns (neighbors [B, k, D], sq_dists [B, k])."""
    B = y.shape[0]
    device = y.device
    best_d = torch.full((B, k), float("inf"), device=device)
    best_i = torch.zeros((B, k), dtype=torch.long, device=device)
    for start in range(0, index.shape[0], chunk_size):
        chunk = index[start:start + chunk_size]
        d2 = torch.cdist(y, chunk) ** 2
        chunk_idx = torch.arange(start, start + chunk.shape[0], device=device).unsqueeze(0).expand(B, -1)
        cat_d = torch.cat([best_d, d2], dim=1)
        cat_i = torch.cat([best_i, chunk_idx], dim=1)
        best_d, top_idx = torch.topk(cat_d, min(k, cat_d.shape[1]), dim=1, largest=False)
        best_i = torch.gather(cat_i, 1, top_idx)
    return index[best_i], best_d


def beta_for_target_qmax(sq_dists, q_target, iters=40, span=1e3, eps=1e-12):
    """Per-sample beta [B] with softmax(-E/beta).max(-1) == q_target, E = d^2/2, by bisection on log beta.

    sq_dists: [B, k] ascending. q_target in (1/k, 1); if unreachable, returns the sharpest beta tried.
    Keeps Var_q(g), and so grad I, nonzero where q would otherwise collapse to one-hot (high D)."""
    E = 0.5 * sq_dists
    E = E - E[:, :1]
    logit = torch.log(torch.as_tensor(q_target / (1.0 - q_target), dtype=E.dtype, device=E.device))
    beta0 = (E[:, 1] / logit.clamp(min=eps)).clamp(min=eps)
    lo = torch.log(beta0 / span)
    hi = torch.log(beta0 * span)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        qm = torch.softmax(-E / mid.exp().unsqueeze(-1), dim=-1).max(-1).values
        too_sharp = qm > q_target
        lo = torch.where(too_sharp, mid, lo)
        hi = torch.where(too_sharp, hi, mid)
    return (0.5 * (lo + hi)).exp()


def bisector_distance(neighbors, sq_dists, eps=1e-8):
    """Distance (d2^2 - d1^2) / (2||Y2 - Y1||) from y to the bisector of its two nearest atoms; [B].

    neighbors [B, k, D], sq_dists [B, k] from ann_query. Rows with duplicate atoms return +inf."""
    Y1, Y2 = neighbors[:, 0], neighbors[:, 1]
    L = (Y2 - Y1).norm(dim=-1)
    d = (sq_dists[:, 1] - sq_dists[:, 0]) / (2 * L.clamp(min=eps))
    return torch.where(L > eps, d, torch.full_like(d, float("inf")))


def fisher_rao_energy(y, neighbors, beta_t, P=None, beta_soft=None):
    """I [B] for y [B, D] against neighbors [B, k, D]; beta_t [B] = sigma_t^2 (not a free parameter).

    P: optional tangent projector (see driver.apply_projector), applied to g before the energy.
    beta_soft: optional per-sample softmax temperature floor [B] (see beta_soft_for); the outer 1/beta_t^2 always
    uses the unfloored beta_t. Returns (I [B], q [B, k], g [B, k, D])."""
    beta_col = beta_t.view(-1, 1)
    if beta_soft is None:
        beta_soft_col = beta_col
    else:
        beta_soft_col = torch.maximum(beta_col, beta_soft.reshape(-1, 1).to(beta_col))
    g = y.unsqueeze(1) - neighbors
    if P is not None:
        g = apply_projector(P, g)
    E = 0.5 * (g ** 2).sum(-1)
    q = torch.softmax(-E / beta_soft_col, dim=-1)
    Eq_g = (q.unsqueeze(-1) * g).sum(1)
    Eq_g2 = (q * (g ** 2).sum(-1)).sum(1)
    I = (Eq_g2 - (Eq_g ** 2).sum(-1)) / beta_t ** 2
    return I, q, g


def beta_soft_for(sq_dists, q_target):
    """Detached beta_for_target_qmax(sq_dists, q_target), or None if q_target is None."""
    if q_target is None:
        return None
    with torch.no_grad():
        return beta_for_target_qmax(sq_dists, q_target).detach()


def make_knn_fr_energy_fn(index, k, beta_t_fn, q_target=None):
    """energy_fn(y) -> I [B]; beta_t_fn() -> sigma_t^2 [B] at the current step."""
    def energy_fn(y):
        with torch.no_grad():
            neighbors, sq_d = ann_query(index, y, k)
        bf = beta_soft_for(sq_d, q_target)
        I, _, _ = fisher_rao_energy(y, neighbors, beta_t_fn(), beta_soft=bf)
        return I
    return energy_fn


def make_knn_projected_energy_fn(index, k, beta_t_fn, q_target=None):
    """energy_fn(y, P) -> I [B]; tangent-projected make_knn_fr_energy_fn."""
    def energy_fn(y, P):
        with torch.no_grad():
            neighbors, sq_d = ann_query(index, y, k)
        bf = beta_soft_for(sq_d, q_target)
        I, _, _ = fisher_rao_energy(y, neighbors, beta_t_fn(), P, beta_soft=bf)
        return I
    return energy_fn
