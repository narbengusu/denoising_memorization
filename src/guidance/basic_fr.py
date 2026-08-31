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


def beta_for_target_qmax(sq_dists, q_target, iters=40, span=1e3, eps=1e-12):
    """Per-sample softmax temperature that holds q_max at a SETPOINT.

    sq_dists: [B, k] ascending squared distances from ann_query. Returns beta
    [B] such that softmax(-E/beta).max(-1) == q_target, with E_i = 1/2 d_i^2.

    This is the adaptive knob that matters at image scale. beta_t is the noise
    schedule and is dimension-free, but the exponent it divides is d^2, which
    is NOT: median d1^2 between CIFAR training images is 433 (0.141 per
    coordinate x D=3072), so at fixed beta_t the posterior over training atoms
    is vastly sharper in 3072 dimensions than on a 2-D toy manifold. Measured
    on the CIFAR subset: q_max = 1.000000 and grad I is EXACTLY zero for every
    sample from roughly progress 0.7 onward. That is not a misconfiguration --
    the true Gibbs posterior really has concentrated -- but it means no
    controller on lambda or eta can do anything there, because they all
    multiply a zero vector. Holding q_max at a setpoint instead keeps
    Var_q(g), and hence the gradient, alive all the way to t=0.

    q_max is monotonically DECREASING in beta (beta -> 0 gives one-hot, beta ->
    inf gives uniform 1/k), so this bisects on log beta. The bracket is
    centered on the exact two-atom solution beta_0 = (E_2 - E_1) / logit(q_target)
    -- correct when the top two atoms dominate, which is the regime this is
    used in -- and widened by `span` in each direction; k > 2 only ever moves
    the answer downward (extra atoms steal mass from q_max), and the bracket
    covers that.

    q_target must lie in (1/k, 1). It can also be UNREACHABLE from above: if
    the two nearest atoms are equidistant, q_max <= ~1/2 no matter how small
    beta gets, and duplicate atoms cap it lower still. Bisection then returns
    the bottom of the bracket, i.e. the sharpest temperature tried, which
    yields the largest achievable q_max -- a graceful floor rather than an
    error, since a sample sitting equidistant between two atoms is exactly the
    non-memorized state guidance is trying to reach anyway.

    Evaluated by the CALLER under no_grad and detached (see
    make_knn_fr_energy_fn): the temperature is a per-step constant, not a
    function of y to backprop through. Differentiating through beta would add
    a d(beta)/d(y) term that has nothing to do with the Fisher-Rao geometry
    the guidance direction is supposed to follow."""
    E = 0.5 * sq_dists
    E = E - E[:, :1]                                    # shift so E_1 = 0; softmax is invariant
    logit = torch.log(torch.as_tensor(q_target / (1.0 - q_target), dtype=E.dtype, device=E.device))
    beta0 = (E[:, 1] / logit.clamp(min=eps)).clamp(min=eps)
    lo = torch.log(beta0 / span)
    hi = torch.log(beta0 * span)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        qm = torch.softmax(-E / mid.exp().unsqueeze(-1), dim=-1).max(-1).values
        too_sharp = qm > q_target                       # q_max too high -> need LARGER beta
        lo = torch.where(too_sharp, mid, lo)
        hi = torch.where(too_sharp, hi, mid)
    return (0.5 * (lo + hi)).exp()


def bisector_distance(neighbors, sq_dists, eps=1e-8):
    """Distance from y to the bisecting hyperplane between its two nearest
    training atoms -- i.e. how far y currently sits from the wall of the cell
    it is in. neighbors: [B, k, D] and sq_dists: [B, k] straight out of
    ann_query (ascending, so index 0/1 are the 1st/2nd nearest). Returns [B].

    For atoms Y1, Y2 the bisector is {y : ||y-Y1|| = ||y-Y2||} and the
    (unsigned, since d2 >= d1 always here) distance to it is

        (d2^2 - d1^2) / (2 ||Y2 - Y1||).

    Exact for basic_fr's cells, not an approximation: fisher_rao_energy's
    E_ij = 1/2||g_ij||^2 carries a beta_t*log N term that is CONSTANT across
    candidates and cancels, so these are plain Voronoi cells, not weighted
    Laguerre ones with offset walls.

    This is the scale a guidance step should be measured against, and the
    reason it is not d1: I peaks ON the cell boundary, so gradient ascent
    walks toward the wall, and a step longer than this lands in the
    NEIGHBORING cell -- closer to a single atom than it started, which is
    collapse accelerated by guidance. Capping at a fraction of this
    asymptotically approaches the ridge without ever crossing it. It also
    does not vanish under memorization the way d1 does: at y = Y1 it equals
    ||Y1 - Y2||/2, so the cap stays wide open exactly where guidance matters
    most, whereas an eta_d*d1 cap would throttle guidance to zero there.

    Duplicate/near-duplicate atoms (||Y2 - Y1|| ~ 0) make the expression 0/0
    and, more to the point, mean there is no wall between them to overshoot;
    those rows return +inf."""
    Y1, Y2 = neighbors[:, 0], neighbors[:, 1]
    L = (Y2 - Y1).norm(dim=-1)
    d = (sq_dists[:, 1] - sq_dists[:, 0]) / (2 * L.clamp(min=eps))
    return torch.where(L > eps, d, torch.full_like(d, float("inf")))


def fisher_rao_energy(y, neighbors, beta_t, P=None, beta_soft=None):
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
    v -> Pv; dispatched via driver.apply_projector. When given, each neighbor
    difference g_ij is projected before anything else -- Fisher-Rao energy is
    defined as variance ALONG TANGENT DIRECTIONS, so restricting g here
    (rather than only projecting a downstream gradient) is what actually
    implements that definition. No extra normalization constant is needed:
    this is a plain sum of squares that already shrinks with the projected
    rank.

    beta_t plays two different roles here: the softmax TEMPERATURE (how
    sharply q concentrates on the nearest neighbor) and the OUTER SCALE
    (1/beta_t^2, which is what should blow up near Laguerre-cell boundaries
    as sigma_t -> 0 -- verified empirically to matter for guidance quality
    late in sampling). Forcing both to the same vanishing beta_t makes q
    saturate to an exact one-hot (verified in float64, not a float32
    precision artifact) well before sigma_t reaches 0, at which point
    Var_q(g) -- and its gradient -- is EXACTLY zero: a genuine dead zone.
    This is NOT primarily a toy-scale phenomenon: measured on a 1000-image
    CIFAR subset with a fixed scalar floor (beta_floor=0.01), q_max =
    1.000000 and ||grad I|| = 0.0 for 64/64 samples from progress ~0.5
    onward -- a scalar floor only helps once it exceeds the energy-gap scale
    (~50 there, i.e. ~3 orders of magnitude larger), which is exactly what a
    fixed constant cannot track per-sample. beta_soft decouples the two
    roles: if given, the softmax uses max(beta_t, beta_soft) as its
    temperature (keeping q from ever fully collapsing, so the gradient stays
    informative), while the outer 1/beta_t^2 division still uses the TRUE,
    unfloored beta_t, so the boundary blow-up behavior late in sampling is
    preserved exactly. beta_soft is a PER-SAMPLE tensor [B]/[B, 1] --
    typically beta_soft_for's output, which solves per sample for the
    temperature that holds q_max at a setpoint (measured range 8.6-189 across
    samples at a single CIFAR step) rather than applying one hand-picked
    constant. beta_soft=None (default) reproduces the original, undecoupled
    behavior.

    E_ij = 1/2||g_ij||^2 (+ beta_t * log N, a constant across all candidates
    that cancels in the softmax below, so it is omitted). Returns (I [B], q
    [B, k], g [B, k, D])."""
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
    """q_target=None -> None (untempered, fisher_rao_energy's default).
    Otherwise the per-sample beta_soft that puts q_max on the setpoint (see
    beta_for_target_qmax), computed under no_grad and detached -- the
    temperature is a per-step constant, not something to backprop through."""
    if q_target is None:
        return None
    with torch.no_grad():
        return beta_for_target_qmax(sq_dists, q_target).detach()


def make_knn_fr_energy_fn(index, k, beta_t_fn, q_target=None):
    """Returns energy_fn(y) -> I [B] for driver.guidance_grad /
    guided_reverse_loop (the AMBIENT, unprojected energy). beta_t_fn() -> [B]
    must return sigma_t^2 at the CURRENT diffusion step -- typically a
    closure reading a mutable t_vec set once per step, same pattern as the
    y_fn.t_vec trick used for the denoiser in the sample scripts. q_target:
    see beta_soft_for/fisher_rao_energy -- None (default) is untempered."""
    def energy_fn(y):
        with torch.no_grad():
            neighbors, sq_d = ann_query(index, y, k)
        bf = beta_soft_for(sq_d, q_target)
        I, _, _ = fisher_rao_energy(y, neighbors, beta_t_fn(), beta_soft=bf)
        return I
    return energy_fn


def make_knn_projected_energy_fn(index, k, beta_t_fn, q_target=None):
    """Returns energy_fn(y, P) -> I [B] for
    driver.projected_guidance_grad/make_projected_guidance_fn -- the tangent-
    restricted counterpart of make_knn_fr_energy_fn. beta_t_fn(), q_target
    same as above."""
    def energy_fn(y, P):
        with torch.no_grad():
            neighbors, sq_d = ann_query(index, y, k)
        bf = beta_soft_for(sq_d, q_target)
        I, _, _ = fisher_rao_energy(y, neighbors, beta_t_fn(), P, beta_soft=bf)
        return I
    return energy_fn
