"""OpenAI's unconditional CIFAR-10 iDDPM (AMG paper Sec. 6.1, 250 DDPM steps; code cloned into external/) and every
method's sampler. Sampler step k is respaced timestep 249 - k; samplers return one {"x", "active"} dict per seed."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from guidance import driver, basic_fr, bayesian_fr, null_control as nc
from baselines import amg_paper, broken_memories as bm
from data.cifar10 import load_cifar10

CLONE = Path(__file__).resolve().parents[2] / "external/improved-diffusion"
REPOSITORY = "https://github.com/openai/improved-diffusion.git"
COMMIT = "1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5"
EXPERIMENT = Path(__file__).resolve().parents[2] / "experiments/cifar10_openai"
MODEL_PATH = EXPERIMENT / "pretrained/cifar10_uncond_50M_500K.pt"
# improved-diffusion README's CIFAR-10 flags; 250 respaced steps as in AMG.
FLAGS = {"image_size": 32, "num_channels": 128, "num_res_blocks": 3, "learn_sigma": True, "dropout": 0.3,
         "diffusion_steps": 4000, "noise_schedule": "cosine", "timestep_respacing": "250"}
NL2_THRESHOLD = 1.4      # memorized if nL2 < this (AMG Sec. 6.1)
FR_SETTINGS = {"k_neighbors": 15, "ell": 1, "window_lo": 0.45, "window_hi": 1.0, "eta": 0.1, "q_target": 0.8,
               "dirichlet_alpha": 0.8, "B_bootstrap": 256}


def ensure_clone():
    if not CLONE.exists():
        subprocess.run(["git", "clone", "--quiet", REPOSITORY, str(CLONE)], check=True)
        subprocess.run(["git", "-C", str(CLONE), "checkout", "--quiet", COMMIT], check=True)
    head = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    assert head.stdout.strip() == COMMIT, f"{CLONE} is not at {COMMIT}"


def load_model(model_path, flags, device):
    """(model, diffusion) for `flags` (overrides of model_and_diffusion_defaults), model in eval mode on `device`."""
    ensure_clone()
    if str(CLONE) not in sys.path:
        sys.path.insert(0, str(CLONE))
    import improved_diffusion.gaussian_diffusion as gd
    from improved_diffusion import script_util

    if device == "mps":
        def extract_float32(arr, timesteps, broadcast_shape):
            res = torch.from_numpy(arr.astype(np.float32)).to(timesteps.device)[timesteps]
            while len(res.shape) < len(broadcast_shape):
                res = res[..., None]
            return res.expand(broadcast_shape)
        gd._extract_into_tensor = extract_float32
    model, diffusion = script_util.create_model_and_diffusion(**{**script_util.model_and_diffusion_defaults(), **flags})
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model.to(device).eval().requires_grad_(False), diffusion


class SeedNoise:
    """Per-seed CPU noise stream: x_T, then one N(0, I) draw per reverse step."""

    def __init__(self, seed, shape=(1, 3, 32, 32)):
        self.generator, self.shape = torch.Generator("cpu").manual_seed(int(seed)), shape

    def draw(self):
        return torch.randn(*self.shape, generator=self.generator)


def load(device, model_path=MODEL_PATH, flags=FLAGS):
    """Model, diffusion and all 50k training images."""
    model, diffusion = load_model(model_path, flags, device)
    train, labels = load_cifar10(split="train")
    X_train_flat = train.reshape(len(train), -1).to(device)
    return dict(device=device, model=model, diffusion=diffusion, wrapped=diffusion._wrap_model(model),
                n_steps=diffusion.num_timesteps, train=train, labels=labels, X_train_flat=X_train_flat,
                train_index=basic_fr.build_flat_index(X_train_flat),
                alphas_cumprod=torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32, device=device))


def pack(x):
    return x.reshape(x.shape[0], -1)


def unpack(z):
    return z.reshape(z.shape[0], 3, 32, 32)


class FixedOutput:
    """Model stand-in returning a precomputed output."""
    def __init__(self, output):
        self.output = output

    def __call__(self, x, ts, **kwargs):
        return self.output


def _timestep(m, i, n):
    return torch.full((n,), i, device=m["device"], dtype=torch.long)


def model_output(m, x, i):
    return m["wrapped"](x, m["diffusion"]._scale_timesteps(_timestep(m, i, len(x))))


def predict_x0(m, x, output, i):
    """Unclipped x0_hat from the model's eps (Eq. 12 of AMG)."""
    return m["diffusion"]._predict_xstart_from_eps(x, _timestep(m, i, len(x)), output[:, :3])


def p_step(m, model_fn, x, i, noise):
    """Ancestral step at respaced timestep i: (x_{i-1}, p_mean_variance output)."""
    out = m["diffusion"].p_mean_variance(model_fn, x, _timestep(m, i, len(x)), clip_denoised=True)
    return out["mean"] + float(i != 0) * torch.exp(0.5 * out["log_variance"]) * noise, out


def nl2_and_ratio(m, y_flat):
    """nL2 (AMG Eq. 8), d1/d2 and nearest index of each row against the training set."""
    values, indices = amg_paper.nearest(y_flat.to(m["device"]), m["X_train_flat"], amg_paper.NEIGHBORS)
    return (values[:, 0] / (amg_paper.ALPHA * values.mean(1))).cpu(), (values[:, 0] / values[:, 1]).cpu(), indices[:, 0].cpu()


def memorization(m, x):
    nl2, ratio, nn = nl2_and_ratio(m, pack(x))
    return [{"nl2": a.item(), "ratio": b.item(), "nn_idx": int(c), "is_memorized": a.item() < NL2_THRESHOLD}
            for a, b, c in zip(nl2, ratio, nn)]


def _streams(seeds):
    return [SeedNoise(s) for s in seeds]


def _draw(m, streams):
    return torch.cat([s.draw() for s in streams]).to(m["device"])


def _cpu_generators(seeds, offset):
    return [torch.Generator("cpu").manual_seed(int(s) + offset) for s in seeds]


def _finish(x, active):
    x = x.clamp(-1, 1).detach().cpu()
    active = torch.stack([torch.as_tensor(a).expand(len(x)) for a in active], 1).cpu()
    return [{"x": x[i:i + 1], "active": active[i]} for i in range(len(x))]


@torch.no_grad()
def unguided(m, seeds):
    streams = _streams(seeds)
    x = _draw(m, streams)
    for i in reversed(range(m["n_steps"])):
        x, _ = p_step(m, m["model"], x, i, _draw(m, streams))
    return _finish(x, [False] * m["n_steps"])


def fr_traces(m, seeds, k=FR_SETTINGS["k_neighbors"], n_snapshots=20):
    """Per-step FR diagnostics (I, q_max, ||grad I||, d1/d2, nn_idx) along the unguided trajectory, for choosing the window.
    Entry j is at progress (j + 1) / n_steps. Returns (unguided endpoints, dict of [len(seeds), n_steps] arrays,
    x0_hat snapshots [len(seeds), n_snapshots, 3, 32, 32])."""
    n_steps, streams = m["n_steps"], _streams(seeds)
    x = _draw(m, streams)
    rec = {key: [] for key in ("I", "q_max", "grad_norm", "ratio", "nn_idx")}
    snap_steps = np.linspace(n_steps / n_snapshots, n_steps, n_snapshots).round().astype(int) - 1
    snaps = []
    for j, i in enumerate(reversed(range(n_steps))):
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            output = model_output(m, x_in, i)
            y = pack(predict_x0(m, x_in, output, i).clamp(-1, 1))
            with torch.no_grad():
                neighbors, _ = basic_fr.ann_query(m["train_index"], y, k)
            I, q, _ = basic_fr.fisher_rao_energy(y, neighbors, (1 - m["alphas_cumprod"][i]).expand(len(y)))
            (grad,) = torch.autograd.grad(I.sum(), x_in)
        with torch.no_grad():
            values, indices = amg_paper.nearest(y, m["X_train_flat"], 2)
            for key, value in (("I", I), ("q_max", q.max(-1).values), ("grad_norm", pack(grad).norm(dim=-1)),
                               ("ratio", values[:, 0] / values[:, 1]), ("nn_idx", indices[:, 0])):
                rec[key].append(value.detach().cpu())
            if j in snap_steps:
                snaps.append(unpack(y).cpu())
            x, _ = p_step(m, FixedOutput(output.detach()), x, i, _draw(m, streams))
    traces = {key: torch.stack(v, 1).numpy() for key, v in rec.items()}
    traces["snapshot_progress"] = (snap_steps + 1) / n_steps
    return _finish(x, [False] * n_steps), traces, torch.stack(snaps, 1)


def _fr_loop(m, seeds, guidance_fn_for, settings):
    streams, state = _streams(seeds), {}

    def base_step(z, step_idx):
        i = m["n_steps"] - step_idx
        state["i"] = i
        with torch.no_grad():
            x_next, _ = p_step(m, m["model"], unpack(z), i, _draw(m, streams))
        return pack(x_next)

    z_T = pack(_draw(m, streams))
    z, I_log = driver.guided_reverse_loop(
        z_T, m["n_steps"], base_step, guidance_fn_for(state),
        progress_lo=settings["window_lo"], progress_hi=settings["window_hi"], ell=settings["ell"], trust_region=settings["eta"])
    active = [False] * m["n_steps"]
    for step_idx, _, n_active in I_log:
        active[step_idx - 1] = n_active > 0
    return _finish(unpack(z), active)


def fr(m, seeds, bayesian=False, settings=FR_SETTINGS):
    """FR (bayesian=False) or Bayesian FR."""
    bootstrap = _cpu_generators(seeds, 500_000)

    def guidance_fn_for(state):
        def y_fn(z):
            x = unpack(z)
            return pack(predict_x0(m, x, model_output(m, x, state["i"]), state["i"]).clamp(-1, 1))

        def energy_fn(y):
            with torch.no_grad():
                neighbors, sq_d = basic_fr.ann_query(m["train_index"], y, settings["k_neighbors"])
            beta_soft = basic_fr.beta_soft_for(sq_d, settings["q_target"])
            sigma2 = (1 - m["alphas_cumprod"][state["i"]]).expand(len(y))
            if bayesian:
                I, _, _ = bayesian_fr.fisher_rao_energy_bb(y, neighbors, sigma2, B=settings["B_bootstrap"],
                                                           dirichlet_alpha=settings["dirichlet_alpha"], beta_soft=beta_soft,
                                                           generator=bootstrap)
            else:
                I, _, _ = basic_fr.fisher_rao_energy(y, neighbors, sigma2, beta_soft=beta_soft)
            return I

        return driver.make_autograd_guidance_fn(y_fn, energy_fn, target_range=None)
    return _fr_loop(m, seeds, guidance_fn_for, settings)


def random(m, seeds, settings=FR_SETTINGS):
    """FR's sampler, eta and window with a uniformly random direction."""
    directions = _cpu_generators(seeds, 99_999)
    return _fr_loop(m, seeds, lambda state: nc.make_null_control_guidance_fn(generator=directions), settings)


def amg(m, seeds, c):
    """AMG, DDPM form (Supp. Eqs. 21-22): mu -= 1{sigma_t > lambda_t} Sigma_t c grad sigma_t, sigma_t = -nL2(x0_hat),
    Sigma_t the learned variance (no guidance at the noiseless last step). c may be scalar or per seed."""
    streams = _streams(seeds)
    c = torch.as_tensor(c, dtype=torch.float32, device=m["device"]).expand(len(seeds))
    x, active = _draw(m, streams), []
    for i in reversed(range(m["n_steps"])):
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            output = model_output(m, x_in, i)
            with torch.no_grad():
                out = m["diffusion"].p_mean_variance(FixedOutput(output.detach()), x_in.detach(), _timestep(m, i, len(x)),
                                                     clip_denoised=True)
            variance = float(i != 0) * torch.exp(out["log_variance"])
            shift, _, on = amg_paper.mean_guidance(x_in, predict_x0(m, x_in, output, i), m["X_train_flat"],
                                                   i / m["n_steps"], c, variance)
        active.append(on.detach())
        with torch.no_grad():
            x = out["mean"] + shift + float(i != 0) * torch.exp(0.5 * out["log_variance"]) * _draw(m, streams)
    return _finish(x, active)


@torch.no_grad()
def _bm_proposal(m, x, i, output, noise):
    """One ancestral step: (proposal, its x0_hat, x0_hat's coefficient sqrt(abar_next) in it, model output there)."""
    proposal, _ = p_step(m, FixedOutput(output), x, i, noise)
    if i == 0:
        return proposal, proposal, 1.0, None
    output_next = model_output(m, proposal, i - 1)
    return proposal, predict_x0(m, proposal, output_next, i - 1), m["alphas_cumprod"][i - 1].sqrt(), output_next


@torch.no_grad()
def norm_traces(m, seeds):
    """Unguided endpoints and the three norms Broken Memories calibrates on, each [len(seeds), n_steps]."""
    streams = _streams(seeds)
    x = _draw(m, streams)
    output = model_output(m, x, m["n_steps"] - 1)
    trace = {name: [] for name in bm.NORMS}
    for i in reversed(range(m["n_steps"])):
        proposal, x0_hat, _, output = _bm_proposal(m, x, i, output, _draw(m, streams))
        for name, value in bm.norms(x, proposal, x0_hat).items():
            trace[name].append(value)
        x = proposal
    return x.clamp(-1, 1).cpu(), {name: torch.stack(v, 1).cpu() for name, v in trace.items()}


@torch.no_grad()
def broken_memories(m, seeds, stats, threshold_scale, mild_steps, gamma):
    streams = _streams(seeds)
    mitigation = bm.Mitigation(stats, len(seeds), m["device"], threshold_scale=threshold_scale, mild_steps=mild_steps, gamma=gamma)
    x = _draw(m, streams)
    output = model_output(m, x, m["n_steps"] - 1)
    active = []
    for step, i in enumerate(reversed(range(m["n_steps"]))):
        proposal, x0_hat, coef, output_next = _bm_proposal(m, x, i, output, _draw(m, streams))
        proposal, on = mitigation.step(step, x, proposal, x0_hat, coef)
        if on.any() and output_next is not None:
            output_next = model_output(m, proposal, i - 1)
        active.append(on)
        x, output = proposal, output_next
    return _finish(x, active)
