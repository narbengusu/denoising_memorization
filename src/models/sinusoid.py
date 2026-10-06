"""Sinusoid toy: 2-D VP-SDE score model on N atoms of y = 0.6 sin(2x), its Euler-Maruyama sampler, every method's
sampler, and the metrics. Near-duplicate atoms (< MERGE_RADIUS apart; each set has a pair 0.011 apart) count as one."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from guidance import driver, basic_fr, bayesian_fr, null_control as nc
from baselines import amg_paper, broken_memories as bm

CHECKPOINTS = Path(__file__).resolve().parents[2] / "experiments/sinusoid/checkpoints"
RATIO_THRESHOLD, MERGE_RADIUS = 1 / 3, 0.05
FR_SETTINGS = {"k_neighbors": 5, "q_target": 0.9, "window_lo": 0.8, "window_hi": 1.0, "ell": 1,
               "B_bootstrap": 256, "dirichlet_alpha": 1.0}


class ScoreNet(nn.Module):
    def __init__(self, d, hidden, depth):
        super().__init__()
        self.input_proj = nn.Linear(d + 1, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2 * depth)])
        self.output_proj = nn.Linear(hidden, d)

    def forward(self, x, t):
        h = F.silu(self.input_proj(torch.cat([x, t[:, None]], -1)))
        for i in range(0, len(self.layers), 2):
            h = h + self.layers[i + 1](F.silu(self.layers[i](h)))
        return self.output_proj(h)


def manifold(u):
    return torch.stack([u, 0.6 * torch.sin(2 * u)], -1)


def curve_geometry(cfg, target):
    """Dense curve (past the sampled range), its arc length, the atoms' arc positions and atom-atom distances."""
    u = torch.linspace(-cfg["u_range"] - 0.5, cfg["u_range"] + 0.5, 40001)
    curve = manifold(u)
    arc = torch.cat([torch.zeros(1), (curve[1:] - curve[:-1]).norm(dim=-1).cumsum(0)])
    atoms = target.cpu()
    return dict(curve=curve, arc=arc, atom_arc=arc[torch.cdist(atoms, curve).argmin(1)], atom_dist=torch.cdist(atoms, atoms))


def load(N, checkpoint_dir=CHECKPOINTS):
    """Trained N-atom model (experiments/sinusoid/01_train_score.ipynb) as a dict."""
    ckpt = torch.load(Path(checkpoint_dir) / f"score_normal_N{N}.pt", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device(cfg["device"])
    target = ckpt["X_target"].to(device)
    net = ScoreNet(cfg["d"], cfg["hidden"], cfg["depth"]).to(device)
    net.load_state_dict(ckpt["net_state"])
    net.requires_grad_(False).eval()
    beta = lambda t: cfg["beta_min"] + t * (cfg["beta_max"] - cfg["beta_min"])
    alpha_bar = lambda t: torch.exp(-(cfg["beta_min"] * t + 0.5 * (cfg["beta_max"] - cfg["beta_min"]) * t ** 2))
    return dict(N=N, cfg=cfg, device=device, target=target, net=net, beta=beta, alpha_bar=alpha_bar,
                n_steps=cfg["n_steps"], dt=cfg["T"] / cfg["n_steps"], geometry=curve_geometry(cfg, target),
                index=basic_fr.build_flat_index(target))


def memorization_ratio(run, x):
    """d1/d2, with d2 the distance to the nearest atom at least MERGE_RADIUS from x's nearest."""
    d = torch.cdist(x.cpu(), run["target"].cpu())
    d1, nearest = d.min(1)
    d2 = d.masked_fill(run["geometry"]["atom_dist"][nearest] < MERGE_RADIUS, float("inf")).min(1).values
    return d1 / d2.clamp_min(1e-8)


def is_memorized(run, x):
    return memorization_ratio(run, x) < RATIO_THRESHOLD


def metrics(run, x):
    """Memorized fraction (ratio and d1 < MERGE_RADIUS), median along-curve offset to the nearest atom, fidelity MSE."""
    x = x.cpu()
    g = run["geometry"]
    foot = torch.cdist(x, g["curve"]).argmin(1)
    along = (g["arc"][foot][:, None] - g["atom_arc"][None]).abs().min(1).values
    return dict(memorized_fraction=is_memorized(run, x).float().mean().item(),
                absolute_memorized_fraction=(torch.cdist(x, run["target"].cpu()).min(1).values < MERGE_RADIUS).float().mean().item(),
                along_curve_offset=along.median().item(),
                fidelity_mse=((x - g["curve"][foot]) ** 2).sum(-1).mean().item())


def _generator(run, seed):
    return torch.Generator(device=run["device"]).manual_seed(seed)


def _t(run, k, n):
    return torch.full((n,), (k + 1) * run["dt"], device=run["device"])


def unguided(run, seed, n, temperature=1.0, traces=False):
    """temperature scales the injected noise; traces=True also returns bm.NORMS traces, each [n, n_steps]."""
    g = _generator(run, seed)
    z = torch.randn(n, 2, generator=g, device=run["device"])
    trace = {k: [] for k in bm.NORMS}
    with torch.no_grad():
        for k in reversed(range(run["n_steps"])):
            t = _t(run, k, n)
            eps = run["net"](z, t); a = run["alpha_bar"](t); b = run["beta"](t)[:, None]
            drift = -0.5 * b * z + b * eps / (1 - a)[:, None].sqrt()
            noise = torch.randn(n, 2, generator=g, device=run["device"]) if k else torch.zeros_like(z)
            proposal = z - drift * run["dt"] + temperature * b.sqrt() * noise * run["dt"] ** 0.5
            if traces:
                tn = torch.full((n,), k * run["dt"], device=run["device"]); an = run["alpha_bar"](tn)[:, None]
                x0 = (proposal - (1 - an).sqrt() * run["net"](proposal, tn)) / an.sqrt()
                for name, value in bm.norms(z, proposal, x0).items():
                    trace[name].append(value)
            z = proposal
    if traces:
        return z.cpu(), {k: torch.stack(v, 1).cpu() for k, v in trace.items()}
    return z.cpu()


def _guided(run, seed, n, eta, guidance_fn_for, settings):
    g = _generator(run, seed)
    hold = {"t": None}

    def base(z, step):
        k = run["n_steps"] - step
        t = torch.full((len(z),), (k + 1) * run["dt"], device=run["device"]); hold["t"] = t
        with torch.no_grad():
            eps = run["net"](z, t); a = run["alpha_bar"](t); b = run["beta"](t)[:, None]
            noise = b.sqrt() * torch.randn(z.shape, generator=g, device=z.device) if k else 0
        return z - (-0.5 * b * z + b * eps / (1 - a)[:, None].sqrt()) * run["dt"] + noise * run["dt"] ** 0.5

    z = torch.randn(n, 2, generator=g, device=run["device"])
    return driver.guided_reverse_loop(z, run["n_steps"], base, guidance_fn_for(hold), progress_lo=settings["window_lo"],
                                      progress_hi=settings["window_hi"], ell=settings["ell"], trust_region=eta)[0].cpu()


def fr(run, seed, n, eta, bayesian=False, settings=FR_SETTINGS):
    """FR (bayesian=False) or Bayesian FR."""
    gb = _generator(run, seed + 1_000_000 + run["N"])

    def guidance_fn_for(hold):
        def denoise(z):
            a = run["alpha_bar"](hold["t"])
            return (z - (1 - a)[:, None].sqrt() * run["net"](z, hold["t"])) / a[:, None].sqrt()
        beta_t = lambda: (1 - run["alpha_bar"](hold["t"])).clamp_min(1e-4)
        energy = (bayesian_fr.make_knn_bayesian_fr_energy_fn(run["index"], settings["k_neighbors"], beta_t,
                                                             B=settings["B_bootstrap"], dirichlet_alpha=settings["dirichlet_alpha"],
                                                             q_target=settings["q_target"], generator=gb)
                  if bayesian else basic_fr.make_knn_fr_energy_fn(run["index"], settings["k_neighbors"], beta_t,
                                                                  q_target=settings["q_target"]))
        return driver.make_autograd_guidance_fn(denoise, energy, target_range=None)
    return _guided(run, seed, n, eta, guidance_fn_for, settings)


def random(run, seed, n, eta, settings=FR_SETTINGS):
    """FR's sampler, eta and window with a uniformly random direction."""
    direction_generator = _generator(run, seed + 3_000_000 + run["N"])
    return _guided(run, seed, n, eta, lambda hold: nc.make_null_control_guidance_fn(generator=direction_generator), settings)


def amg(run, seed, n, c3, window=1.0):
    """AMG (Eqs. 10, 17): eps += c3 sqrt(1 - abar) grad sigma_t while sigma_t > lambda_t, sigma_t = -nL2(x0_hat).
    window < 1 (not in the paper) guides only steps with t/T <= window."""
    g = _generator(run, seed)
    z = torch.randn(n, 2, generator=g, device=run["device"])
    for k in reversed(range(run["n_steps"])):
        t = _t(run, k, n); a = run["alpha_bar"](t); b = run["beta"](t)[:, None]
        z_in = z.detach().requires_grad_(True)
        with torch.enable_grad():
            eps = run["net"](z_in, t)
            x0 = (z_in - (1 - a)[:, None].sqrt() * eps) / a[:, None].sqrt()
            level = (k + 1) * run["dt"] / run["cfg"]["T"]
            shift, _, _ = amg_paper.eps_guidance(z_in, x0, run["target"], level, c3 if window >= 1 or level <= window else 0, (1 - a).sqrt())
        eps = eps.detach() + shift
        noise = torch.randn(n, 2, generator=g, device=run["device"]) if k else torch.zeros_like(z)
        z = (z - (-0.5 * b * z + b * eps / (1 - a)[:, None].sqrt()) * run["dt"] + b.sqrt() * noise * run["dt"] ** 0.5).detach()
    return z.cpu()


def calibrate_bm(run, n=512):
    """Broken Memories' stability regions from n unguided, non-memorized trajectories."""
    x, traces = unguided(run, 110 + run["N"], n, traces=True)
    normal = ~is_memorized(run, x)
    return bm.calibrate({name: v[normal] for name, v in traces.items()})


def broken_memories(run, stats, seed, n, threshold_scale, duration, gamma=3):
    g = _generator(run, seed)
    z = torch.randn(n, 2, generator=g, device=run["device"])
    mitigation = bm.Mitigation(stats, n, run["device"], threshold_scale=threshold_scale, mild_steps=duration, gamma=gamma)
    with torch.no_grad():
        for step, k in enumerate(reversed(range(run["n_steps"]))):
            t = _t(run, k, n); a = run["alpha_bar"](t); b = run["beta"](t)[:, None]
            eps = run["net"](z, t)
            noise = torch.randn(n, 2, generator=g, device=run["device"]) if k else torch.zeros_like(z)
            proposal = z - (-0.5 * b * z + b * eps / (1 - a)[:, None].sqrt()) * run["dt"] + b.sqrt() * noise * run["dt"] ** 0.5
            tn = torch.full((n,), k * run["dt"], device=run["device"]); an = run["alpha_bar"](tn)[:, None]
            x0 = (proposal - (1 - an).sqrt() * run["net"](proposal, tn)) / an.sqrt()
            z, _ = mitigation.step(step, z, proposal, x0, an.sqrt())
    return z.cpu()
