"""Model- and energy-agnostic guided reverse sampling loop.

All callables operate on a flat state z_t [B, D]; the energy comes from basic_fr or bayesian_fr.
"""
import torch


def gate_in_range(I, target_range):
    """True where I lies in target_range=(lo, hi); all True if None."""
    if target_range is None:
        return torch.ones_like(I, dtype=torch.bool)
    lo, hi = target_range
    return (I >= lo) & (I <= hi)


def _clip_grad_norm(g, clip_norm, eps=1e-8):
    """Cap each row's norm at clip_norm (scalar or per-row [B]/[B, 1] tensor)."""
    if clip_norm is None:
        return g
    gn = g.norm(dim=-1, keepdim=True).clamp(min=eps)
    if torch.is_tensor(clip_norm):
        clip_norm = clip_norm.reshape(gn.shape)
    scale = (clip_norm / gn).clamp(max=1.0)
    return g * scale


def _resolve_clip_norm(grad_clip, z_t):
    """grad_clip: None, a scalar, or a callable z_t -> per-row cap [B]/[B, 1] (evaluated under no_grad)."""
    if not callable(grad_clip):
        return grad_clip
    with torch.no_grad():
        return grad_clip(z_t)


def guidance_grad(y_fn, z_t, energy_fn, target_range=None, denoise=True, grad_clip=None):
    """Ambient guidance: exact autograd of I(D_t(z_t)) w.r.t. z_t, zeroed outside target_range.

    denoise=False evaluates energy_fn at z_t itself instead of the denoised endpoint (ablation).
    grad_clip: see _resolve_clip_norm; needed with guidance_correction_lambda, which does not normalize.
    Returns (grad [B, D], I [B], in_range [B] bool)."""
    with torch.enable_grad():
        z_g = z_t.detach().requires_grad_(True)
        y = y_fn(z_g) if denoise else z_g
        I = energy_fn(y)
        in_range = gate_in_range(I, target_range)

        grad = torch.zeros_like(z_t)
        if torch.any(in_range):
            (g_full,) = torch.autograd.grad(I.sum(), z_g)
            g_full = _clip_grad_norm(g_full, _resolve_clip_norm(grad_clip, z_t))
            mask = in_range.to(g_full.dtype).view(-1, *([1] * (g_full.dim() - 1)))
            grad = g_full * mask
    return grad, I.detach(), in_range


def make_autograd_guidance_fn(y_fn, energy_fn, target_range=None, denoise=True, grad_clip=None):
    """Wrap guidance_grad as guidance_fn(z_t) -> (grad, I, in_range)."""
    def guidance_fn(z_t):
        return guidance_grad(y_fn, z_t, energy_fn, target_range, denoise, grad_clip)
    return guidance_fn


def solve_tangent_z_step(J, g, ridge_rel=0.1):
    """Ridge-regularized solve of J dz = g: dz = (J^T J + ridge_rel * tr(J^T J)/D * I)^-1 J^T g.

    g is a y-space displacement, so it is pulled back with J^-1, not J^T. Dense D x D; toy scale only."""
    Jt = J.transpose(-1, -2)
    A = torch.bmm(Jt, J)
    D = J.shape[-1]
    scale = A.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp(min=1e-12)
    eye = torch.eye(D, device=J.device, dtype=J.dtype).unsqueeze(0)
    A_reg = A + (ridge_rel * scale)[:, None, None] * eye
    b = torch.bmm(Jt, g.unsqueeze(-1)).squeeze(-1)
    return torch.linalg.solve(A_reg, b.unsqueeze(-1)).squeeze(-1)


def projected_guidance_grad(y_fn, z_t, energy_fn, projector_fn, target_range=None,
                             denoiser_jacobian_fn=None, ridge_rel=0.1, grad_clip=None):
    """Tangent-projected guidance; energy_fn(y, P) -> I [B], with P built from a detached y.

    denoiser_jacobian_fn(z_t) -> J [B, D, D] = dy/dz_t maps grad_y I to z_t via solve_tangent_z_step;
    None uses grad_y I directly (assumes dy/dz_t = I). grad_clip is applied to the final z_t-space step.
    Returns (grad [B, D], I [B], in_range [B] bool)."""
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
                    J = denoiser_jacobian_fn(z_t)
                    g = solve_tangent_z_step(J, g, ridge_rel)
            g = _clip_grad_norm(g, _resolve_clip_norm(grad_clip, z_t))
            mask = in_range.to(g.dtype).view(-1, *([1] * (g.dim() - 1)))
            grad = g * mask
    return grad, I.detach(), in_range


def make_projected_guidance_fn(y_fn, energy_fn, projector_fn, target_range=None,
                                denoiser_jacobian_fn=None, ridge_rel=0.1, grad_clip=None):
    """Wrap projected_guidance_grad as guidance_fn(z_t) -> (grad, I, in_range)."""
    def guidance_fn(z_t):
        return projected_guidance_grad(y_fn, z_t, energy_fn, projector_fn, target_range,
                                        denoiser_jacobian_fn, ridge_rel, grad_clip)
    return guidance_fn


def projected_guidance_grad_chained(y_fn, z_t, energy_fn, projector_fn, target_range=None, grad_clip=None):
    """Tangent-projected guidance by autograd through y_fn (a J^T v product), avoiding the unstable J solve.

    Returns (grad [B, D], I [B], in_range [B] bool)."""
    with torch.no_grad():
        P = projector_fn(y_fn(z_t))
    with torch.enable_grad():
        z_g = z_t.detach().requires_grad_(True)
        y = y_fn(z_g)
        I = energy_fn(y, P)
        in_range = gate_in_range(I, target_range)

        grad = torch.zeros_like(z_t)
        if torch.any(in_range):
            (g_full,) = torch.autograd.grad(I.sum(), z_g)
            g_full = _clip_grad_norm(g_full, _resolve_clip_norm(grad_clip, z_t))
            mask = in_range.to(g_full.dtype).view(-1, *([1] * (g_full.dim() - 1)))
            grad = g_full * mask
    return grad, I.detach(), in_range


def make_projected_guidance_fn_chained(y_fn, energy_fn, projector_fn, target_range=None, grad_clip=None):
    """Wrap projected_guidance_grad_chained as guidance_fn(z_t) -> (grad, I, in_range)."""
    def guidance_fn(z_t):
        return projected_guidance_grad_chained(y_fn, z_t, energy_fn, projector_fn, target_range, grad_clip)
    return guidance_fn


def _eigh_robust(M):
    """eigh in float64 with non-finite entries sanitized and escalating diagonal jitter on LinAlgError."""
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
    """P = V_k V_k^T [B, D, D] over the k smallest-|eigenvalue| directions of symmetric M (largest=True: k largest)."""
    lam, V = _eigh_robust(M)
    D = M.shape[-1]
    idx = lam.abs().topk(k, largest=largest).indices
    V_k = V.gather(2, idx.unsqueeze(1).expand(-1, D, -1))
    return torch.bmm(V_k, V_k.transpose(-1, -2))


def make_projector_fn(jacobian_matrix_fn, k, largest=False):
    """projector_fn(y) -> P [B, D, D] from jacobian_matrix_fn(y) -> M [B, D, D]; k = manifold dimension. Dense."""
    def projector_fn(y):
        return eig_tangent_projector(jacobian_matrix_fn(y), k, largest=largest)
    return projector_fn


def apply_projector(P, v):
    """Apply P (callable v -> Pv, or dense symmetric [B, D, D]) to v [B, ..., D]."""
    if callable(P):
        return P(v)
    orig_shape = v.shape
    v_flat = v.reshape(v.shape[0], -1, v.shape[-1])
    out = torch.bmm(v_flat, P.transpose(-1, -2))
    return out.reshape(orig_shape)


def guidance_window(step_idx, n_steps, progress_lo, progress_hi, ell):
    """True every ell-th step while step_idx / n_steps lies in [progress_lo, progress_hi].

    progress is the fraction of the sampling loop elapsed (0 = pure noise), i.e. t decreases as progress grows."""
    if step_idx < progress_lo * n_steps:
        return False
    if step_idx > progress_hi * n_steps:
        return False
    return step_idx % ell == 0


def _resolve_eta(eta, step_idx):
    """eta: a scalar or a callable eta_fn(step_idx), evaluated under no_grad."""
    if not callable(eta):
        return eta
    with torch.no_grad():
        return eta(step_idx)


def guidance_correction(grad, z, z_next, trust_region, step_idx, eps=1e-8):
    """Correction eta * ||z_next - z|| * grad / ||grad||; zero where ||grad|| <= eps (exact dead zones of I).

    eta is dimensionless but dataset-dependent; calibrate once per dataset."""
    eta = _resolve_eta(trust_region, step_idx)
    gn = grad.norm(dim=-1, keepdim=True)
    base = (z_next - z).norm(dim=-1, keepdim=True)
    corr = eta * base * grad / gn.clamp(min=eps)
    return torch.where(gn > eps, corr, torch.zeros_like(grad))


def _resolve_step_scale(step_scale, step_idx):
    """step_scale: a scalar or a callable step_scale_fn(step_idx) (e.g. sigma_t**2), evaluated under no_grad."""
    if not callable(step_scale):
        return step_scale
    with torch.no_grad():
        return step_scale(step_idx)


def guidance_correction_lambda(grad, lam, step_scale, step_idx):
    """Non-adaptive correction lam * step_scale(step_idx) * grad, without normalization."""
    scale = _resolve_step_scale(step_scale, step_idx)
    if torch.is_tensor(scale) and scale.dim() > 0:
        scale = scale.view(-1, *([1] * (grad.dim() - 1)))
    return lam * scale * grad


def guided_reverse_loop(z_init, n_steps, base_step_fn, guidance_fn,
                         progress_lo, progress_hi, ell, trust_region,
                         measure_fn=None, measure_steps=None, hist_out=None,
                         lam=None, step_scale=None):
    """Reverse loop: base_step_fn(z_t, step_idx) -> z_{t-1}, plus guidance_fn(z_t) -> (grad, I, in_range) in the window.

    trust_region: eta (scalar or callable) for guidance_correction; ignored when lam is given
    (then guidance_correction_lambda with step_scale is used).
    measure_fn(z_t, step_idx) -> I [B]: recorded into hist_out[step_idx] at measure_steps, independent of the window.
    Returns (z_0, I_log: list of (step_idx, mean I, n_active)).
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
                if lam is not None:
                    z_next = z_next + guidance_correction_lambda(grad, lam, step_scale, step_idx)
                else:
                    z_next = z_next + guidance_correction(grad, z, z_next, trust_region, step_idx)
            I_log.append((step_idx, I.mean().item(), int(in_range.sum().item())))
        z = z_next
    return z, I_log
