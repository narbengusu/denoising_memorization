"""Energy function for the empirical score-divergence guidance variant --
for use with driver.guided_reverse_loop.

I_beta(y) = trace(grad_y s_theta(y, t)) + d/sigma_t^2, where the trace (the
divergence of the model's own score field, i.e. the Laplacian of its learned
log-density) is estimated with r Hutchinson probes. Unlike basic_fr.py,
this needs no training-atom index at all -- the "is y near a memorized mode"
signal comes entirely from the local curvature of the model's own score,
which spikes at sharp (near-delta) density peaks.

Two ways to get grad_{x_t} I from here, trading exactness for memory:

  make_score_div_energy_fn      -- exact, via driver.make_autograd_guidance_fn.
                                    Needs create_graph=True through the trace
                                    estimate so the OUTER grad_{x_t} can differentiate
                                    it again -- i.e. a double-backward through the
                                    whole network, r times, with both graphs alive
                                    at once. This is what OOMs at large batch sizes:
                                    verified empirically on an H100 (79GB) with
                                    B=500, a 9-layer EGNN, r=4 -- CUDA OOM trying to
                                    allocate ~400MB on top of ~78GB already in use.

  make_score_div_fd_guidance_fn -- finite-difference (SPSA) outer gradient, no
                                    second backward graph ever created -- see its
                                    docstring. Prefer this at real (500+) batch
                                    sizes; use the exact path only if it fits.
"""
import torch

from .driver import _clip_grad_norm, gate_in_range, apply_projector


def hutchinson_trace(score_fn, y, r, P=None):
    """score_fn: y -> s(y), same shape as y, differentiable w.r.t. y (y must
    still carry the graph back to whatever it depends on, so a caller can
    backprop through this trace estimate again -- do NOT detach y here).

    Standard reverse-mode double-backprop trick (as in FFJORD): for a probe v
    with iid mean-0 unit-variance entries, E_v[v^T J^T v] = trace(J) for any
    square J, so a single vjp per probe suffices -- no forward-mode JVP
    needed.

    P: optional tangent projector at y -- either a dense [B, D, D] tensor or
    a callable v -> Pv (see basic_fr.fisher_rao_energy's P docstring; same
    driver.apply_projector dispatch). When given, each probe is restricted to
    the tangent subspace via u = P @ v before the vjp: since P is symmetric
    idempotent and Cov(v) = I, E[u^T J u] = E[v^T P J P v] = trace(P J P) =
    trace(J P) -- the divergence restricted to tangent directions, with no
    eigenbasis needed explicitly here (P already encodes it). Returns a [B]
    trace estimate."""
    s = score_fn(y)
    trace_hat = 0.0
    for _ in range(r):
        v = torch.randn_like(y)
        if P is not None:
            v = apply_projector(P, v)
        (Jt_v,) = torch.autograd.grad(s, y, grad_outputs=v, create_graph=True, retain_graph=True)
        trace_hat = trace_hat + (v * Jt_v).flatten(1).sum(-1)
    return trace_hat / r


def make_score_div_energy_fn(score_fn, sigma2_fn, d_fn, r):
    """Returns energy_fn(y) -> I [B] for driver.make_autograd_guidance_fn
    (the EXACT, double-backward path -- see module docstring for its memory cost).

    score_fn(y) -> s(y)   the model's score at y, at whatever diffusion time
                          the caller is currently stepping through (typically
                          a closure reading a mutable t_vec set once per step,
                          same pattern as y_fn.t_vec in the sample scripts --
                          energy_fn's signature only ever takes y).
    sigma2_fn() -> [B]    sigma_t^2 at the current time.
    d_fn() -> [B]         true (unpadded) dimensionality of y, per sample --
                          padded/masked entries must NOT be counted here, since
                          the model forces their score to identically zero
                          (so they contribute exactly 0 to the trace estimate
                          too; including them in d would bias the correction).
    r: number of Hutchinson probes.
    """
    def energy_fn(y):
        div_hat = hutchinson_trace(score_fn, y, r)
        return div_hat + d_fn() / sigma2_fn()
    return energy_fn


def make_score_div_projected_energy_fn(score_fn, sigma2_fn, k_fn, r):
    """Returns energy_fn(y, P) -> I [B] for
    driver.projected_guidance_grad/make_projected_guidance_fn -- the tangent-
    restricted counterpart of make_score_div_energy_fn.

    k_fn() -> [B] or scalar     tangent-subspace dimension (rank of P), NOT
                                the full ambient d_fn() -- once the trace
                                itself is restricted to k dimensions via P,
                                the flat-Gaussian correction term must scale
                                with k, not d, or the two terms are on
                                mismatched subspaces.
    score_fn, sigma2_fn, r: same as make_score_div_energy_fn."""
    def energy_fn(y, P):
        div_hat = hutchinson_trace(score_fn, y, r, P=P)
        return div_hat + k_fn() / sigma2_fn()
    return energy_fn


def hutchinson_trace_value(score_fn, y, r):
    """Same trace estimate as hutchinson_trace, but for use where the caller
    only needs I(y) as a NUMBER, not something to differentiate further: each
    probe's vjp uses an ordinary (create_graph=False) backward, so the graph
    for that probe is freed immediately instead of being kept alive for a
    second differentiation. y is detached and re-wrapped as its own leaf here
    (unlike hutchinson_trace, which requires y to still carry a graph back to
    x_t) -- this is the building block for the finite-difference guidance_fn
    below, which gets grad_{x_t} I from repeated forward evaluations of I at
    perturbed x_t rather than from autograd through I itself."""
    with torch.enable_grad():
        y_req = y.detach().requires_grad_(True)
        s = score_fn(y_req)
        trace_hat = 0.0
        for _ in range(r):
            v = torch.randn_like(y_req)
            (Jt_v,) = torch.autograd.grad(s, y_req, grad_outputs=v, retain_graph=True)
            trace_hat = trace_hat + (v * Jt_v).flatten(1).sum(-1)
    return (trace_hat / r).detach()


def make_score_div_fd_guidance_fn(y_fn, score_fn, sigma2_fn, d_fn, r,
                                   n_dirs=4, eps=1e-2, target_range=None, grad_clip=None):
    """Returns guidance_fn(z_t) -> (grad, I, in_range) directly (unlike
    make_score_div_energy_fn, this is NOT meant to go through
    driver.make_autograd_guidance_fn -- it builds its own gradient
    without autograd on the outer step at all).

    Estimates grad_{z_t} I(D_t(z_t)) via simultaneous-perturbation stochastic
    approximation (SPSA): for n_dirs random unit directions u, central
    differences (I(z+eps*u) - I(z-eps*u)) / (2*eps) give a directional
    derivative, and averaging u * (directional derivative) over directions
    gives an unbiased gradient estimate of a smoothed version of I -- all of
    it via plain forward passes through y_fn + the r-probe trace estimate
    (hutchinson_trace_value), never differentiating THAT estimate a second
    time. This trades the exact method's peak memory (two nested backward
    graphs through the whole network, alive simultaneously -- what OOMs at
    large batch sizes) for 2*n_dirs extra forward-plus-single-backward energy
    evaluations per guidance step, each freed immediately.

    y_fn(z_t) -> y   same denoiser used elsewhere; does not need to be
                     differentiable w.r.t. z_t here (no autograd on this path).
    score_fn, sigma2_fn, d_fn, r: same as make_score_div_energy_fn.
    n_dirs: number of SPSA probe directions per guidance step.
    eps: finite-difference step size in z_t-space.
    """
    def energy_value_fn(y):
        div_hat = hutchinson_trace_value(score_fn, y, r)
        return div_hat + d_fn() / sigma2_fn()

    def guidance_fn(z_t):
        with torch.no_grad():
            y0 = y_fn(z_t)
            I0 = energy_value_fn(y0)
            in_range = gate_in_range(I0, target_range)

            grad = torch.zeros_like(z_t)
            if torch.any(in_range):
                for _ in range(n_dirs):
                    u = torch.randn_like(z_t)
                    u = u / u.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    I_pos = energy_value_fn(y_fn(z_t + eps * u))
                    I_neg = energy_value_fn(y_fn(z_t - eps * u))
                    coeff = ((I_pos - I_neg) / (2 * eps)).view(-1, *([1] * (u.dim() - 1)))
                    grad = grad + coeff * u
                grad = grad / n_dirs
                grad = _clip_grad_norm(grad, grad_clip)
                mask = in_range.to(grad.dtype).view(-1, *([1] * (grad.dim() - 1)))
                grad = grad * mask
        return grad, I0, in_range
    return guidance_fn
