"""Broken Memories (Huang et al., 2026), Algorithms 1-2, for samplers that predict a clean image.

Calibrated on non-memorized unguided runs: per-step mean/std of ||x_t||, ||delta_t|| = ||proposal - x_t||, ||x0_hat||.
Detection: first step with |z(delta)| and |z(x0)| > tau_mild; strong if |z(x0)| > tau_strong, else mild.
Mitigation (|z(delta)| > tau_mild and strong, or mild before step k): rescale x0_hat to the mean norm
(proposal += x0_coef * (x0' - x0_hat); x0_coef = sqrt(abar_next) for VP, 1 for EDM); strong only: clamp ||delta_t||
to mean +- gamma std; then clamp ||x_t||.
Paper setting (tau_mild, tau_strong, gamma, k) = (3, 14, 3, 10) at 50 steps; designed for CFG in Stable Diffusion.
"""
import torch

NORMS = ("x_norm", "delta_norm", "x0_norm")
TAU_MILD, TAU_STRONG, GAMMA, MILD_STEPS = 3.0, 14.0, 3.0, 10


def _flat_norm(v):
    return v.reshape(len(v), -1).norm(dim=-1)


def norms(x, proposal, x0_hat):
    """The three calibrated norms of one step, each of shape [B]."""
    return {"x_norm": _flat_norm(proposal), "delta_norm": _flat_norm(proposal - x), "x0_norm": _flat_norm(x0_hat)}


def calibrate(traces, std_floor=1e-5):
    """traces: {name: [n_trajectories, n_steps]} from non-memorized unguided runs -> {name: {"mean", "std"}}."""
    return {name: {"mean": v.mean(0), "std": v.std(0).clamp(min=std_floor)} for name, v in traces.items()}


def _rescale_to(v, target):
    return v * (target / _flat_norm(v).clamp(min=1e-8)).view(-1, *([1] * (v.ndim - 1)))


def _project_norm(v, lo, hi):
    norm = _flat_norm(v).clamp(min=1e-8)
    return v * (norm.clamp(min=lo.clamp(min=0), max=hi) / norm).view(-1, *([1] * (v.ndim - 1)))


class Mitigation:
    """Per-sample detection state and the mitigation rule, for a batch of B trajectories."""

    def __init__(self, stats, batch, device, threshold_scale=1.0, mild_steps=MILD_STEPS, gamma=GAMMA,
                 tau_mild=TAU_MILD, tau_strong=TAU_STRONG):
        self.stats = {name: {k: v.to(device) for k, v in s.items()} for name, s in stats.items()}
        self.tau_mild, self.tau_strong = tau_mild * threshold_scale, tau_strong * threshold_scale
        self.mild_steps, self.gamma = mild_steps, gamma
        self.kind = torch.zeros(batch, dtype=torch.long, device=device)   # 0 none, 1 mild, 2 strong

    def _z(self, value, name, step):
        return ((value - self.stats[name]["mean"][step]) / self.stats[name]["std"][step]).abs()

    def _band(self, name, step):
        mu, sd = self.stats[name]["mean"][step], self.stats[name]["std"][step]
        return mu - self.gamma * sd, mu + self.gamma * sd

    def step(self, step, x, proposal, x0_hat, x0_coef):
        """Returns (proposal, active [B]) for the proposal of sampler step `step` (0-based)."""
        n = norms(x, proposal, x0_hat)
        s_delta, s_x0 = self._z(n["delta_norm"], "delta_norm", step), self._z(n["x0_norm"], "x0_norm", step)
        first = (self.kind == 0) & (s_delta > self.tau_mild) & (s_x0 > self.tau_mild)
        self.kind[first] = torch.where(s_x0[first] > self.tau_strong, 2, 1)
        active = (s_delta > self.tau_mild) & ((self.kind == 2) | ((self.kind == 1) & (step < self.mild_steps)))
        if not active.any():
            return proposal, active
        mask = active.view(-1, *([1] * (x.ndim - 1)))
        target = self.stats["x0_norm"]["mean"][step].expand(len(x))
        mitigated = proposal + x0_coef * (_rescale_to(x0_hat, target) - x0_hat)
        strong = (active & (self.kind == 2)).view_as(mask)
        mitigated = torch.where(strong, x + _project_norm(mitigated - x, *self._band("delta_norm", step)), mitigated)
        mitigated = _project_norm(mitigated, *self._band("x_norm", step))
        return torch.where(mask, mitigated, proposal), active
