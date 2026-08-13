"""Guided diffusion sampling driver -- model- and energy-method-agnostic.

Implements the shared control flow behind all FR-guided sampling algorithm
variants (basic_fr's k-NN Fisher-Rao energy, bayesian_fr's Dirichlet-bootstrap
posterior variant, and score_divergence's empirical score-divergence energy):
a base reverse step, spliced with a guidance correction at every `ell`-th step
inside a t* window, gated by an optional target range on the guidance energy
I(y), and optionally projected onto an estimated tangent subspace via
`jacobian_fn` (see eig_tangent_projector / make_jacobian_fn below).

The state being diffused, the base reverse step, the denoiser, and the energy
function are all passed in as callables operating on a single flat tensor
`z_t` of shape [B, D] -- any model (SO(n) toy MLP, QM9 EGNN, ...) plugs in by
writing pack/unpack + step/denoise wrappers around its own state
representation, and one of `basic_fr.py`, `bayesian_fr.py`, or
`score_divergence.py` supplies the energy function. Nothing here is QM9- or
SO(n)-specific, and nothing here is specific to any one guidance method.
"""
import torch


def _clip_grad_norm(g, clip_norm, eps=1e-8):
    """Cap each row's norm at clip_norm; leave sub-threshold rows unchanged.

    I(y) is unbounded for both guidance energies here (unlike a bounded field
    like a Voronoi margin), so the raw gradient can spike by orders of
    magnitude near a genuine cluster boundary or a sharp score singularity --
    left unclipped this reliably diverges the sampler (verified empirically: a
    single spike at one guidance step sends x to NaN within a couple of
    subsequent steps). A soft cap preserves direction and small-gradient
    behavior away from interfaces while preventing blowup.

    clip_norm may be a python scalar (fixed cap, broadcasts to every row) or
    a per-row tensor of shape [B] or [B, 1] (e.g. an adaptive, x/t-dependent
    cap from _resolve_clip_norm below) -- both divide elementwise into gn."""
    if clip_norm is None:
        return g
    gn = g.norm(dim=-1, keepdim=True).clamp(min=eps)
    if torch.is_tensor(clip_norm):
        clip_norm = clip_norm.reshape(gn.shape)
    scale = (clip_norm / gn).clamp(max=1.0)
    return g * scale


def _resolve_clip_norm(grad_clip, z_t):
    """grad_clip is either None, a fixed scalar, or a callable adaptive_clip_fn(z_t)
    -> per-row cap [B] or [B, 1] -- e.g. ||score(z_t, t)|| at the current (x, t),
    so the guidance correction is never allowed to outweigh the model's own drift
    at that point rather than being bounded by one hand-picked constant for every
    x and every t. Callable is evaluated under no_grad -- the cap itself is just a
    magnitude, not something to backprop through."""
    if not callable(grad_clip):
        return grad_clip
    with torch.no_grad():
        return grad_clip(z_t)


def gate_in_range(I, target_range):
    """Shared second-gating-condition helper: True everywhere if
    target_range is None, else True only where I falls in [lo, hi]."""
    if target_range is None:
        return torch.ones_like(I, dtype=torch.bool)
    lo, hi = target_range
    return (I >= lo) & (I <= hi)


def guidance_grad(y_fn, z_t, energy_fn, target_range=None, grad_clip=None):
    """Compute grad_{z_t} I(D_t(z_t)) via EXACT autograd through y_fn (the
    denoiser) and energy_fn (either algorithm's I(y)). If target_range=(lo,
    hi) is given, guidance is zeroed out for batch elements whose I falls
    outside it (the algorithm's second gating condition); target_range=None
    applies guidance to every element whenever called (the caller still
    enforces the window/period gate).

    grad_clip: None (no clipping), a fixed scalar, or a callable
    adaptive_clip_fn(z_t) -> per-row cap [B]/[B, 1] -- see _resolve_clip_norm.

    This is the AMBIENT (no manifold information) baseline -- classic
    classifier-guidance-style exact gradient of I(D_t(z_t)) w.r.t. z_t. When
    a tangent projector is available, use projected_guidance_grad instead:
    chaining this gradient through y_fn and then reprojecting with a
    projector built at y would conflate the denoiser's own (unrelated)
    Jacobian dy/dz_t with the data manifold's tangent geometry.

    Returns (grad [same shape as z_t], I.detach() [B], in_range [B] bool).

    Runs under its own `torch.enable_grad()` so it works whether or not the
    surrounding sampling loop is wrapped in `torch.no_grad()` (the base steps
    of a reverse loop typically are; this guidance step is the exception).
    energy_fn may itself use double-backward internally (the score-divergence
    variant's exact mode does) for that method, DOUBLE-BACKWARD THROUGH
    THE WHOLE NETWORK is what it costs to get an exact gradient here, which
    can OOM at large batch sizes; score_divergence.make_score_div_fd_guidance_fn
    is a finite-difference alternative that never needs a second backward
    graph, at the cost of a noisier gradient estimate. See its docstring."""
    with torch.enable_grad():
        z_g = z_t.detach().requires_grad_(True)
        y = y_fn(z_g)
        I = energy_fn(y)
        in_range = gate_in_range(I, target_range)

        grad = torch.zeros_like(z_t)
        if torch.any(in_range):
            (g_full,) = torch.autograd.grad(I.sum(), z_g)
            g_full = _clip_grad_norm(g_full, _resolve_clip_norm(grad_clip, z_t))
            mask = in_range.to(g_full.dtype).view(-1, *([1] * (g_full.dim() - 1)))
            grad = g_full * mask
    return grad, I.detach(), in_range


def make_autograd_guidance_fn(y_fn, energy_fn, target_range=None, grad_clip=None):
    """Adapter: wraps guidance_grad into the guidance_fn(z_t) -> (grad, I,
    in_range) signature guided_reverse_loop expects, for guidance methods
    that use exact autograd (as opposed to score_divergence's
    finite-difference alternative, which builds a guidance_fn directly)."""
    def guidance_fn(z_t):
        return guidance_grad(y_fn, z_t, energy_fn, target_range, grad_clip)
    return guidance_fn


def projected_guidance_grad(y_fn, z_t, energy_fn, projector_fn, target_range=None, grad_clip=None):
    """Compute the tangent-projected guidance correction directly in y-space:
    Jacobian(y) * grad_y I(y, P), NOT grad_z I(D_t(z_t)) reprojected after the
    fact. energy_fn(y, P) -> I [B] must itself use P to restrict the energy
    to tangent directions (e.g. basic_fr.make_knn_projected_energy_fn,
    score_divergence.make_score_div_projected_energy_fn) -- both the energy
    and its gradient are then tangent-restricted by construction, so there is
    no incoherent reprojection of an ambient gradient that passed through the
    denoiser's own unrelated Jacobian.

    y_fn is evaluated under no_grad and re-wrapped as its own leaf: no
    backward pass through the denoiser is built at all here, only through
    energy_fn -- for basic_fr this means guidance needs no backprop through
    the score network whatsoever.

    grad_clip is applied AFTER projection (the correction is already
    tangential by this point, so the clip bounds that directly, rather than
    bounding an ambient vector that then gets shrunk again by projection).

    Returns (grad [same shape as z_t], I.detach() [B], in_range [B] bool)."""
    with torch.no_grad():
        y = y_fn(z_t)
    P = projector_fn(y)
    with torch.enable_grad():
        y_leaf = y.detach().requires_grad_(True)
        I = energy_fn(y_leaf, P)
        in_range = gate_in_range(I, target_range)

        grad = torch.zeros_like(z_t)
        if torch.any(in_range):
            (g,) = torch.autograd.grad(I.sum(), y_leaf)
            g = _clip_grad_norm(g, _resolve_clip_norm(grad_clip, z_t))
            mask = in_range.to(g.dtype).view(-1, *([1] * (g.dim() - 1)))
            grad = g * mask
    return grad, I.detach(), in_range


def make_projected_guidance_fn(y_fn, energy_fn, projector_fn, target_range=None, grad_clip=None):
    """Adapter: wraps projected_guidance_grad into the guidance_fn(z_t) ->
    (grad, I, in_range) signature guided_reverse_loop expects."""
    def guidance_fn(z_t):
        return projected_guidance_grad(y_fn, z_t, energy_fn, projector_fn, target_range, grad_clip)
    return guidance_fn


def _eigh_robust(M):
    """torch.linalg.eigh (LAPACK syevd, divide-and-conquer) can fail to converge on
    ill-conditioned, near-degenerate, or non-finite matrices -- observed in practice on a
    JacNet-estimated M near t->0, where dividing by sigma_t^2 blows up M's magnitude/conditioning
    (the same regime where basic_fr's energy separately blows up, see the diagnostic panel) and
    can push JacNet's own output to NaN/Inf. Sanitizes non-finite entries first (a persistent
    convergence failure even under escalating jitter, as opposed to an isolated one-off, is a
    strong sign the input wasn't finite to begin with), then retries in float64 with
    progressively larger diagonal jitter -- a uniform eigenvalue shift, so it doesn't change
    which directions rank top-k, only breaks numerical degeneracies severe enough that no fixed
    jitter scale is guaranteed to resolve them in one try."""
    M = torch.nan_to_num(M, nan=0.0, posinf=1e6, neginf=-1e6)
    D = M.shape[-1]
    scale = M.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1).clamp(min=1.0).double()
    eye = torch.eye(D, device=M.device, dtype=torch.float64).unsqueeze(0)
    M64 = M.double()
    last_err = None
    for jitter_rel in (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0):
        try:
            lam, V = torch.linalg.eigh(M64 + jitter_rel * scale[:, None, None] * eye)
            return lam.to(M.dtype), V.to(M.dtype)
        except torch.linalg.LinAlgError as e:
            last_err = e
    raise last_err


def eig_tangent_projector(M, k, largest=False):
    """M: [B, D, D] symmetric (a Jacobian/metric estimate). Keeps the top-k
    eigenvectors by |eigenvalue| -- largest=False (default) selects the
    smallest-|eigenvalue| directions, i.e. the estimated TANGENT subspace of
    dimension k (small |eigenvalue| = the score barely curves along that
    direction, consistent with it lying along the data manifold); largest=True
    selects the k NORMAL directions instead, for diagnostics that compare
    guidance strength tangentially vs. normally. Returns P [B, D, D] = V_k @ V_k^T."""
    lam, V = _eigh_robust(M)
    D = M.shape[-1]
    idx = lam.abs().topk(k, largest=largest).indices
    V_k = V.gather(2, idx.unsqueeze(1).expand(-1, D, -1))
    return torch.bmm(V_k, V_k.transpose(-1, -2))


def make_projector_fn(jacobian_matrix_fn, k, largest=False):
    """jacobian_matrix_fn(y) -> M [B, D, D]: a per-sample Jacobian/metric
    estimate (e.g. a trained JacNet's rescaled score Jacobian). Returns a
    projector_fn(y) -> P [B, D, D], the top-k eigendirections of M, for use
    with projected_guidance_grad/make_projected_guidance_fn (P feeds BOTH the
    energy and its gradient there, not just the gradient). k is the
    manifold-dimension ablation knob -- how many tangent directions to keep.

    Dense: forms a [B, D, D] matrix, fine at toy/SO(n) scale (D up to a few
    hundred) but NOT at image scale -- see make_lowrank_normal_projector_fn
    for the D=thousands case."""
    def projector_fn(y):
        return eig_tangent_projector(jacobian_matrix_fn(y), k, largest=largest)
    return projector_fn


def apply_projector(P, v):
    """Applies a projector to v: [B, ..., D]. P may be either a callable
    v -> Pv (the operator-based path -- see make_lowrank_normal_projector_fn),
    or a dense [B, D, D] tensor (the legacy path -- eig_tangent_projector/
    make_projector_fn). Every caller of a projector inside this package
    (basic_fr.fisher_rao_energy, score_divergence.hutchinson_trace) goes
    through this dispatch instead of applying P directly, so dense- and
    operator-based projectors are interchangeable everywhere."""
    if callable(P):
        return P(v)
    orig_shape = v.shape
    v_flat = v.reshape(v.shape[0], -1, v.shape[-1])
    out = torch.bmm(v_flat, P.transpose(-1, -2))   # P symmetric, so this == bmm(P, v)
    return out.reshape(orig_shape)


def make_lowrank_normal_projector_fn(u_fn, r_prime):
    """u_fn(y) -> U [B, D, R]: a per-sample low-rank factor (e.g. a trained
    low-rank JacNet's output) such that M ~ U U^T approximates the rescaled
    score Jacobian restricted to its R dominant (highest-|eigenvalue|,
    "normal"/high-curvature) directions -- the only structure a low-rank M
    can represent, unlike eig_tangent_projector's dense path, which can
    resolve genuine near-zero-eigenvalue ("tangent") directions directly.

    Consequently this returns the ORTHOGONAL-COMPLEMENT projector -- remove
    JacNet's top-r' identified normal directions rather than keep a small-k
    tangent subset (which would be numerically arbitrary here: with M
    rank-R, the "smallest-eigenvalue" space is a massively degenerate
    (D-R)-dim null space with no preferred orientation to select k of).

    Q_r' = U's top-r' left-singular vectors, found via one [B, D, R]
    economy SVD (R small, e.g. 8) -- never forms a dense [B, D, D] matrix.
    Returns projector_fn(y) -> callable v -> v - Q_r'(Q_r'^T v)."""
    def projector_fn(y):
        U = u_fn(y)
        Q, _, _ = torch.linalg.svd(U, full_matrices=False)   # Q: [B, D, R], descending singular value
        Qr = Q[:, :, :r_prime]

        def apply(v):
            orig_shape = v.shape
            v_flat = v.reshape(v.shape[0], -1, v.shape[-1])
            coeff = torch.bmm(v_flat, Qr)
            proj_component = torch.bmm(coeff, Qr.transpose(-1, -2))
            return (v_flat - proj_component).reshape(orig_shape)
        return apply
    return projector_fn


def guidance_window(step_idx, n_steps, progress_lo, progress_hi, ell):
    """step_idx: number of reverse steps completed so far (1 = right after the
    first step, n_steps = the last). progress_lo/progress_hi are fractions of
    the SAMPLING LOOP elapsed (progress = step_idx/n_steps: 0.0 = just
    started, x_t is still pure noise; 1.0 = finished, x_t is clean data) --
    deliberately NOT named after the schedule's own diffusion-time variable
    `t`, which moves the OPPOSITE direction (t=T at the start of sampling,
    t=0 at the end): "guide from 20% to 90% of the way through sampling" is
    progress_lo=0.2, progress_hi=0.9 directly, which corresponds to t
    decreasing from ~0.8*T down to ~0.1*T, not to t=0.2 or t=0.9.

    Guidance fires while step_idx/n_steps is in [progress_lo, progress_hi],
    every `ell`-th such step."""
    if step_idx < progress_lo * n_steps:
        return False
    if step_idx > progress_hi * n_steps:
        return False
    return step_idx % ell == 0


def guided_reverse_loop(z_init, n_steps, base_step_fn, guidance_fn,
                         progress_lo, progress_hi, ell, lam, step_scale_fn,
                         measure_fn=None, measure_steps=None, hist_out=None):
    """Run the full T..1 reverse loop with guidance spliced in -- shared by
    both algorithms; only `guidance_fn` differs between them (and between the
    exact-autograd and finite-difference variants of the same algorithm).

    base_step_fn(z_t, step_idx) -> z_{t-1}       (the model's own unguided reverse step)
    guidance_fn(z_t) -> (grad, I [B], in_range [B])   built by
        driver.make_autograd_guidance_fn(...) or
        score_divergence.make_score_div_fd_guidance_fn(...) (or the basic_fr /
        exact score-divergence equivalents) -- already closes over y_fn,
        energy_fn, target_range, grad_clip, and jacobian_fn, so it only ever
        takes z_t.
    step_scale_fn(step_idx) -> float or tensor   (the algorithm's a_t(x_t) term
                                                scaling the guidance correction;
                                                pass a schedule, e.g. sigma_t^2)
    progress_lo, progress_hi: see guidance_window -- fractions of the sampling
                loop elapsed, NOT the schedule's t (progress_hi=1.0 guides
                through the very end of sampling; progress_lo=0.0 guides from
                the very start).

    measure_fn(z_t, step_idx) -> I [B], measure_steps (iterable of step_idx),
    hist_out (dict, mutated in place as hist_out[step_idx] = I.cpu()): an
    optional, independent diagnostic hook for recording the FULL per-sample
    I(y) distribution at specific steps (e.g. steps nearest a set of t
    fractions), regardless of the guidance window/period gate above -- unlike
    I_log's I.mean() over only the steps guidance actually fired on. All three
    default to None/empty, so existing callers that don't pass them are
    unaffected and the return signature is unchanged.

    Returns (z_0, I_log: list of (step_idx, I.mean().item(), n_active)).
    """
    measure_steps = set(measure_steps) if measure_steps else set()
    z = z_init
    I_log = []
    for step_idx in range(1, n_steps + 1):
        if measure_fn is not None and hist_out is not None and step_idx in measure_steps:
            hist_out[step_idx] = measure_fn(z, step_idx).detach().cpu()
        z_next = base_step_fn(z, step_idx)
        if guidance_window(step_idx, n_steps, progress_lo, progress_hi, ell):
            grad, I, in_range = guidance_fn(z)
            if torch.any(in_range):
                z_next = z_next + lam * step_scale_fn(step_idx) * grad
            I_log.append((step_idx, I.mean().item(), int(in_range.sum().item())))
        z = z_next
    return z, I_log
