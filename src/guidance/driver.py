"""Guided diffusion sampling driver -- model- and energy-method-agnostic.

Implements the shared control flow behind the FR-guided sampling algorithm
(basic_fr's k-NN Fisher-Rao energy, and bayesian_fr's Dirichlet-bootstrap
posterior variant): a base reverse step, spliced with a guidance correction
at every `ell`-th step inside a t* window, gated by an optional target range
on the guidance energy I(y), and optionally projected onto an estimated
tangent subspace via `jacobian_fn` (see eig_tangent_projector / make_jacobian_fn
below).

The state being diffused, the base reverse step, the denoiser, and the energy
function are all passed in as callables operating on a single flat tensor
`z_t` of shape [B, D] -- any model (SO(n) toy MLP, QM9 EGNN, ...) plugs in by
writing pack/unpack + step/denoise wrappers around its own state
representation, and one of `basic_fr.py` or `bayesian_fr.py` supplies the
energy function. Nothing here is QM9- or SO(n)-specific, and nothing here is
specific to any one guidance method.
"""
import torch


def gate_in_range(I, target_range):
    """Shared second-gating-condition helper: True everywhere if
    target_range is None, else True only where I falls in [lo, hi]."""
    if target_range is None:
        return torch.ones_like(I, dtype=torch.bool)
    lo, hi = target_range
    return (I >= lo) & (I <= hi)


def guidance_grad(y_fn, z_t, energy_fn, target_range=None, denoise=True):
    """Compute grad_{z_t} I(D_t(z_t)) via EXACT autograd through y_fn (the
    denoiser) and energy_fn (the algorithm's I(y)). If target_range=(lo,
    hi) is given, guidance is zeroed out for batch elements whose I falls
    outside it (the algorithm's second gating condition); target_range=None
    applies guidance to every element whenever called (the caller still
    enforces the window/period gate).

    denoise: True (default) evaluates energy_fn at y = y_fn(z_g), the
    denoiser's expected-endpoint estimate D_t(z_t) -- this is the algorithm
    as designed. False skips the denoiser entirely and evaluates energy_fn
    directly at z_g (the current, still-noisy point) -- an ablation to
    isolate whether routing guidance through the expected endpoint matters,
    vs. just reacting to I at x_t itself. Note energy_fn's own scale/meaning
    may not be comparable between the two settings (e.g. a k-NN energy built
    against a clean-data index sees an out-of-distribution query when handed
    a noisy z_t) -- treat denoise=True/False as different quantities, not
    directly comparable magnitudes.

    This is the AMBIENT (no manifold information) baseline -- classic
    classifier-guidance-style exact gradient of I(D_t(z_t)) w.r.t. z_t. When
    a tangent projector is available, use projected_guidance_grad instead:
    chaining this gradient through y_fn and then reprojecting with a
    projector built at y would conflate the denoiser's own (unrelated)
    Jacobian dy/dz_t with the data manifold's tangent geometry.

    Returns (grad [same shape as z_t], I.detach() [B], in_range [B] bool).

    Runs under its own `torch.enable_grad()` so it works whether or not the
    surrounding sampling loop is wrapped in `torch.no_grad()` (the base steps
    of a reverse loop typically are; this guidance step is the exception)."""
    with torch.enable_grad():
        z_g = z_t.detach().requires_grad_(True)
        y = y_fn(z_g) if denoise else z_g
        I = energy_fn(y)
        in_range = gate_in_range(I, target_range)

        grad = torch.zeros_like(z_t)
        if torch.any(in_range):
            (g_full,) = torch.autograd.grad(I.sum(), z_g)
            mask = in_range.to(g_full.dtype).view(-1, *([1] * (g_full.dim() - 1)))
            grad = g_full * mask
    return grad, I.detach(), in_range


def make_autograd_guidance_fn(y_fn, energy_fn, target_range=None, denoise=True):
    """Adapter: wraps guidance_grad into the guidance_fn(z_t) -> (grad, I,
    in_range) signature guided_reverse_loop expects.

    denoise: see guidance_grad -- True (default) evaluates energy_fn at the
    denoised endpoint D_t(z_t); False evaluates it directly at z_t."""
    def guidance_fn(z_t):
        return guidance_grad(y_fn, z_t, energy_fn, target_range, denoise)
    return guidance_fn


def projected_guidance_grad(y_fn, z_t, energy_fn, projector_fn, target_range=None,
                             denoiser_jacobian_fn=None):
    """Compute the tangent-projected guidance correction. `P` is built from a
    detached snapshot of `y` (the tangent DIRECTIONS should reflect the
    manifold/energy geometry at y, not get perturbed by whatever Jacobian
    machinery is used elsewhere) -- energy_fn(y, P) -> I [B] must itself use
    P to restrict the energy to tangent directions (e.g.
    basic_fr.make_knn_projected_energy_fn).

    denoiser_jacobian_fn=None (default): grad_y I(y, P) is computed in
    y-space and used AS-IS as the z_t-space correction -- implicitly assumes
    dy/dz_t = identity. Cheap (no Jacobian of any kind needed). But verified
    empirically (toy_manifold debugging session) that the denoiser is
    strongly non-identity at high t, and this default can point the
    correction in the WRONG direction for the majority of samples during the
    noisiest ~20% of a reverse trajectory (median cosine similarity to the
    true grad_z_t I(D_t(z_t), P) was negative there) despite having a
    plausible-looking magnitude.

    denoiser_jacobian_fn(z_t) -> J [B, D, D] (dy/dz_t, exact or a cheap
    approximation -- e.g. jacrev/vmap over y_fn at toy/SO(n) scale where D is
    small, or a dedicated small network at image/molecule scale where a full
    backward pass through the real denoiser would be the expensive
    alternative): when given, the returned gradient is J^T @ grad_y I(y, P),
    the correctly Jacobian-mapped z_t-space correction -- mathematically
    equivalent to backprop-ing all the way through the denoiser, but without
    ever running its (possibly large) backward pass, since J is supplied
    directly. This is what should actually be used; None only exists so
    callers that haven't wired up a jacobian_fn yet keep their old behavior.

    Returns (grad [same shape as z_t], I.detach() [B], in_range [B] bool)."""
    with torch.no_grad():
        y_snapshot = y_fn(z_t)
    P = projector_fn(y_snapshot)
    with torch.enable_grad():
        y_leaf = y_snapshot.detach().requires_grad_(True)
        I = energy_fn(y_leaf, P)
        in_range = gate_in_range(I, target_range)

        grad = torch.zeros_like(z_t)
        if torch.any(in_range):
            (g,) = torch.autograd.grad(I.sum(), y_leaf)
            if denoiser_jacobian_fn is not None:
                with torch.no_grad():
                    J = denoiser_jacobian_fn(z_t)   # [B, D, D], dy/dz_t
                g = torch.bmm(J.transpose(-1, -2), g.unsqueeze(-1)).squeeze(-1)
            mask = in_range.to(g.dtype).view(-1, *([1] * (g.dim() - 1)))
            grad = g * mask
    return grad, I.detach(), in_range


def make_projected_guidance_fn(y_fn, energy_fn, projector_fn, target_range=None,
                                denoiser_jacobian_fn=None):
    """Adapter: wraps projected_guidance_grad into the guidance_fn(z_t) ->
    (grad, I, in_range) signature guided_reverse_loop expects."""
    def guidance_fn(z_t):
        return projected_guidance_grad(y_fn, z_t, energy_fn, projector_fn, target_range,
                                        denoiser_jacobian_fn)
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
    hundred) but not tractable at image scale (D=thousands)."""
    def projector_fn(y):
        return eig_tangent_projector(jacobian_matrix_fn(y), k, largest=largest)
    return projector_fn


def apply_projector(P, v):
    """Applies a projector to v: [B, ..., D]. P may be either a callable
    v -> Pv, or a dense [B, D, D] tensor (the path used here --
    eig_tangent_projector/make_projector_fn). Every caller of a projector
    inside this package (basic_fr.fisher_rao_energy) goes through this
    dispatch instead of applying P directly, so dense- and operator-based
    projectors are interchangeable everywhere."""
    if callable(P):
        return P(v)
    orig_shape = v.shape
    v_flat = v.reshape(v.shape[0], -1, v.shape[-1])
    out = torch.bmm(v_flat, P.transpose(-1, -2))   # P symmetric, so this == bmm(P, v)
    return out.reshape(orig_shape)


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


def _resolve_eta(eta, step_idx):
    """eta is either a fixed scalar (fine as-is) or a callable eta_fn(step_idx)
    -> float/tensor -- e.g. a beta(t) or beta(t)^-1 schedule, for trying a
    functional guidance-strength profile instead of one constant for the whole
    trajectory. Evaluated under no_grad since it's a magnitude, not something
    to backprop through."""
    if not callable(eta):
        return eta
    with torch.no_grad():
        return eta(step_idx)


def guidance_correction(grad, z, z_next, trust_region, step_idx, eps=1e-8):
    """Turn a raw guidance gradient into the correction actually added to
    z_next: eta * ||z_next - z|| * grad/||grad||. eta (trust_region: a float
    or a callable eta_fn(step_idx)) is dimensionless -- "move this fraction of
    what the base sampler itself moves this step" -- so it is invariant to t,
    to sigma, and to gradient spikes. NOT invariant to ambient dimension or to
    the dataset: the toy manifold (d=2) needs eta=1.0 where CIFAR-10
    (D=3072) needs eta=0.1 -- calibrate once per dataset on a geometric
    ladder, then freeze it. Two attempts to derive it from first principles
    (a cumulative-displacement budget; matching the sampler's own
    deterministic drift) both underestimated the toy manifold's working value
    by 40-800x, because eta behaves like a threshold, not a scale match.

    The ||grad|| > eps guard is load-bearing, not defensive boilerplate:
    fisher_rao_energy has genuine dead zones where Var_q(g) -- and hence grad
    -- is EXACTLY zero (one-hot q collapse; see its docstring). Normalizing
    there would turn float noise into a full-magnitude step in an arbitrary
    direction, which looks exactly like the off-manifold collapse failure mode
    and would be easy to misattribute to eta being too large."""
    eta = _resolve_eta(trust_region, step_idx)
    gn = grad.norm(dim=-1, keepdim=True)
    base = (z_next - z).norm(dim=-1, keepdim=True)
    corr = eta * base * grad / gn.clamp(min=eps)
    return torch.where(gn > eps, corr, torch.zeros_like(grad))


def guided_reverse_loop(z_init, n_steps, base_step_fn, guidance_fn,
                         progress_lo, progress_hi, ell, trust_region,
                         measure_fn=None, measure_steps=None, hist_out=None):
    """Run the full T..1 reverse loop with guidance spliced in -- shared by
    both algorithms; only `guidance_fn` differs between them.

    base_step_fn(z_t, step_idx) -> z_{t-1}       (the model's own unguided reverse step)
    guidance_fn(z_t) -> (grad, I [B], in_range [B])   built by
        driver.make_autograd_guidance_fn(...) or make_projected_guidance_fn(...)
        -- already closes over y_fn, energy_fn, and target_range, so it only
        ever takes z_t.
    trust_region: float or a callable eta_fn(step_idx) -- the relative
        parameterization, eta * ||base step|| * grad/||grad||. See
        guidance_correction.
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
                z_next = z_next + guidance_correction(grad, z, z_next, trust_region, step_idx)
            I_log.append((step_idx, I.mean().item(), int(in_range.sum().item())))
        z = z_next
    return z, I_log
