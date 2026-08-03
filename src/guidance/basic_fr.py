"""Energy function for Algorithm 1 (Fisher-Rao Energy Guided Diffusion
Sampling, k-NN variant, the "Basic" algorithm) -- for use with driver.guided_reverse_loop.

I_beta(y) = (1/beta^2) [ E_q||g||^2 - ||E_q g||^2 ],  g_i = y - Y_i,
q = softmax(-E_i / beta) over the k retrieved training points Y_i.
"""
import torch

from .driver import apply_projector


def build_flat_index(Y_flat):
    """Y_flat: [N, D] training points. Kept as a thin wrapper so the exact
    brute-force search below can later be swapped for a real ANN library
    (faiss, scann, ...) without touching the guidance math."""
    return Y_flat.detach()


def ann_query(index, y, k, chunk_size=20_000):
    """Exact k-NN via chunked brute-force search (O(N) per query, chunked to
    bound memory) -- stands in for the paper's O(d log N) ANN index. y: [B, D].
    Returns (neighbors [B, k, D], sq_dists [B, k])."""
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


def fisher_rao_energy(y, neighbors, beta_t, P=None):
    """y: [B, D], neighbors: [B, k, D] (the k retrieved training points).
    beta_t: [B] -- NOT a free hyperparameter; the algorithm's Require line
    lists only the noise schedule sigma_t (beta_t never appears there
    separately), so beta_t = sigma_t * sigma_t^T = sigma_t^2 (the schedule's
    own noise variance at the current step, isotropic here). Passing a fixed
    constant instead is wrong: it stops the softmax bandwidth from sharpening
    as sigma_t -> 0 late in sampling, which is exactly the "local two-atom
    limit" behavior the energy is supposed to have near clean data.

    P: optional tangent projector at y -- either a dense [B, D, D] tensor
    (e.g. from driver.make_projector_fn or an analytic tangent) or a callable
    v -> Pv (e.g. from driver.make_lowrank_normal_projector_fn, at image
    scale where a dense D x D matrix is intractable); dispatched via
    driver.apply_projector. When given, each neighbor difference g_ij is
    projected before anything else -- Fisher-Rao energy is defined as
    variance ALONG TANGENT DIRECTIONS, so restricting g here (rather than
    only projecting a downstream gradient) is what actually implements that
    definition. No extra normalization constant is needed: unlike
    score-divergence's flat trace correction, this is a plain sum of squares
    that already shrinks with the projected rank.

    E_ij = 1/2||g_ij||^2 (+ beta_t * log N, a constant across all candidates
    that cancels in the softmax below, so it is omitted). Returns (I [B], q
    [B, k], g [B, k, D])."""
    beta_col = beta_t.view(-1, 1)
    g = y.unsqueeze(1) - neighbors
    if P is not None:
        g = apply_projector(P, g)
    E = 0.5 * (g ** 2).sum(-1)
    q = torch.softmax(-E / beta_col, dim=-1)
    Eq_g = (q.unsqueeze(-1) * g).sum(1)
    Eq_g2 = (q * (g ** 2).sum(-1)).sum(1)
    I = (Eq_g2 - (Eq_g ** 2).sum(-1)) / beta_t ** 2
    return I, q, g


def make_knn_fr_energy_fn(index, k, beta_t_fn):
    """Returns energy_fn(y) -> I [B] for driver.guidance_grad /
    guided_reverse_loop (the AMBIENT, unprojected energy). beta_t_fn() -> [B]
    must return sigma_t^2 at the CURRENT diffusion step -- typically a
    closure reading a mutable t_vec set once per step, same pattern as the
    y_fn.t_vec trick used for the denoiser in the sample scripts."""
    def energy_fn(y):
        with torch.no_grad():
            neighbors, _ = ann_query(index, y, k)
        I, _, _ = fisher_rao_energy(y, neighbors, beta_t_fn())
        return I
    return energy_fn


def make_knn_projected_energy_fn(index, k, beta_t_fn):
    """Returns energy_fn(y, P) -> I [B] for
    driver.projected_guidance_grad/make_projected_guidance_fn -- the tangent-
    restricted counterpart of make_knn_fr_energy_fn. beta_t_fn() same as
    above."""
    def energy_fn(y, P):
        with torch.no_grad():
            neighbors, _ = ann_query(index, y, k)
        I, _, _ = fisher_rao_energy(y, neighbors, beta_t_fn(), P)
        return I
    return energy_fn
