"""CelebA-HQ latent EDM fine-tuned on 200 SD-VAE latents (4x32x32), its deterministic 32-step Heun sampler, and every
method's sampler. Samplers return {"z": [1, 4, 32, 32] on CPU, "active": [n_steps] steps where the method intervened}."""
import gc
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import UNet2DModel, AutoencoderKL

from guidance import driver, basic_fr, bayesian_fr, null_control as nc
from baselines import amg_paper, broken_memories as bm

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/celeba_hq"
N_STEPS, SIGMA_MIN, SIGMA_MAX, RHO = 32, 0.002, 80.0, 7.0
RATIO_THRESHOLD = 1 / 3          # memorized if d1/d2 < this
PIXEL_RES = 256                  # resolution the latents were encoded at
FR_SETTINGS = {"k_neighbors": 15, "ell": 1, "window_lo": 0.35, "window_hi": 1.0, "eta": 0.2, "q_target": 0.8,
               "dirichlet_alpha": 0.2, "B_bootstrap": 256}


def load(device, with_vae=True, checkpoints=EXPERIMENT / "checkpoints", vae_dir=EXPERIMENT / "pretrained/sd-vae-ft-mse",
         images_dir=ROOT / "data/celeba_hq/images"):
    """Fine-tuned model, its 200 training latents, the training image files and (with_vae) the SD VAE."""
    torch.backends.cuda.enable_cudnn_sdp(False)
    ckpt = torch.load(Path(checkpoints) / "celeba_hq_edm_finetuned.pt", map_location="cpu", weights_only=False)
    sigma_data = ckpt["config"]["sigma_data"]
    net = UNet2DModel(
        sample_size=32, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(128, 256, 384, 384),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )
    net.load_state_dict(ckpt["model_ema"])
    del ckpt
    latents = torch.load(Path(checkpoints) / "celeba_hq_latents.pt", map_location="cpu", weights_only=False)["latents"].float()
    finetune_idx = np.load(Path(checkpoints) / "celeba_hq_finetune_train_idx.npy")
    X_train = latents[finetune_idx].to(device)
    del latents
    gc.collect()
    X_train_flat = X_train.reshape(len(X_train), -1)
    i = torch.arange(N_STEPS, dtype=torch.float64)
    sigmas = (SIGMA_MAX ** (1 / RHO) + i / (N_STEPS - 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
    m = dict(device=device, net=net.to(device).requires_grad_(False).eval(), sigma_data=sigma_data,
             sigmas=torch.cat([sigmas, torch.zeros(1)]).float().to(device), X_train=X_train, X_train_flat=X_train_flat,
             train_index=basic_fr.build_flat_index(X_train_flat), finetune_idx=finetune_idx,
             images_dir=Path(images_dir), train_files=sorted(os.listdir(images_dir)), pixels_flat=None,
             ckpt_path=Path(checkpoints) / "celeba_hq_edm_finetuned.pt")
    if with_vae:
        m["vae"] = AutoencoderKL.from_pretrained(vae_dir).to(device).eval().requires_grad_(False)
    return m


def pack(x):
    return x.reshape(x.shape[0], -1)


def unpack(z):
    return z.reshape(z.shape[0], 4, 32, 32)


def precond(m, x, sigma):
    """EDM-preconditioned denoiser D(x; sigma)."""
    sigma = sigma.view(-1, 1, 1, 1)
    sd = m["sigma_data"]
    c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
    c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
    c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
    return c_skip * x + c_out * m["net"](c_in * x, 0.25 * sigma.log().flatten()).sample


def z_init(m, seed):
    generator = torch.Generator(device=m["device"] if m["device"] != "mps" else "cpu").manual_seed(seed)
    return torch.randn(1, 4, 32, 32, generator=generator).to(m["device"]) * m["sigmas"][0]


def heun_step(m, x, step, denoised, denoise_next=None):
    """One Heun step from sigmas[step] to sigmas[step + 1], given D(x) at sigmas[step]."""
    s_cur, s_next = m["sigmas"][step], m["sigmas"][step + 1]
    d_cur = (x - denoised) / s_cur
    x_next = x + (s_next - s_cur) * d_cur
    if s_next > 0:
        denoised_next = denoise_next(x_next) if denoise_next else precond(m, x_next, s_next.expand(1))
        x_next = x + (s_next - s_cur) * 0.5 * (d_cur + (x_next - denoised_next) / s_next)
    return x_next


def memorization(m, z):
    top2 = torch.cdist(pack(z.to(m["device"])), m["X_train_flat"]).topk(2, largest=False)
    d1, d2 = top2.values[0, 0].item(), top2.values[0, 1].item()
    return {"d1": d1, "d2": d2, "ratio": d1 / d2, "nn_idx": int(top2.indices[0, 0]), "is_memorized": d1 / d2 < RATIO_THRESHOLD}


def load_train_image(m, row):
    """Fine-tuning row -> uint8 [3, PIXEL_RES, PIXEL_RES]. The latent pool is [unflipped, flipped]."""
    pool_idx = int(m["finetune_idx"][row])
    files = m["train_files"]
    image = Image.open(m["images_dir"] / files[pool_idx % len(files)]).convert("RGB")
    if pool_idx >= len(files):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    image = image.resize((PIXEL_RES, PIXEL_RES), Image.LANCZOS)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)


def training_pixels(m):
    """Fine-tuning images in [-1, 1], flattened and cached (AMG's pixel-space search, Supp. D)."""
    if m["pixels_flat"] is None:
        m["pixels_flat"] = (torch.stack([load_train_image(m, i) for i in range(len(m["finetune_idx"]))]).float()
                            .div(127.5).sub(1).reshape(len(m["finetune_idx"]), -1).to(m["device"]))
    return m["pixels_flat"]


@torch.no_grad()
def decode(m, z):
    image = m["vae"].decode(unpack(z.to(m["device"])) / m["vae"].config.scaling_factor).sample
    return (((image.clamp(-1, 1) + 1) / 2) * 255).round().to(torch.uint8)[0].cpu()


def _finish(x, active):
    return {"z": x.detach().cpu(), "active": torch.tensor(active)}


@torch.no_grad()
def unguided(m, seed):
    x = z_init(m, seed)
    for step in range(N_STEPS):
        x = heun_step(m, x, step, precond(m, x, m["sigmas"][step].expand(1)))
    return _finish(x, [False] * N_STEPS)


def _fr_loop(m, seed, guidance_fn_for, settings):
    state = {}

    def base_step(z, step_idx):
        step = step_idx - 1
        state["s"] = m["sigmas"][step]
        x = unpack(z)
        return pack(heun_step(m, x, step, precond(m, x, m["sigmas"][step].expand(1))))

    z, I_log = driver.guided_reverse_loop(
        pack(z_init(m, seed)), N_STEPS, base_step, guidance_fn_for(state),
        progress_lo=settings["window_lo"], progress_hi=settings["window_hi"], ell=settings["ell"], trust_region=settings["eta"])
    active = [False] * N_STEPS
    for step_idx, _, n_active in I_log:
        active[step_idx - 1] = n_active > 0
    return _finish(unpack(z), active)


def fr(m, seed, bayesian=False, settings=FR_SETTINGS):
    """FR (bayesian=False) or Bayesian FR."""
    generator = torch.Generator(device=m["device"]).manual_seed(seed)

    def guidance_fn_for(state):
        def y_fn(z):
            return pack(precond(m, unpack(z), state["s"].expand(1)))

        def energy_fn(y):
            with torch.no_grad():
                neighbors, sq_d = basic_fr.ann_query(m["train_index"], y, settings["k_neighbors"])
            beta_soft = basic_fr.beta_soft_for(sq_d, settings["q_target"])
            sigma2 = (state["s"] ** 2).expand(1)
            if bayesian:
                I, _, _ = bayesian_fr.fisher_rao_energy_bb(y, neighbors, sigma2, B=settings["B_bootstrap"],
                                                           dirichlet_alpha=settings["dirichlet_alpha"], beta_soft=beta_soft,
                                                           generator=generator)
            else:
                I, _, _ = basic_fr.fisher_rao_energy(y, neighbors, sigma2, beta_soft=beta_soft)
            return I

        return driver.make_autograd_guidance_fn(y_fn, energy_fn, target_range=None)
    return _fr_loop(m, seed, guidance_fn_for, settings)


def random(m, seed, settings=FR_SETTINGS):
    """FR's sampler, eta and window with a uniformly random direction."""
    directions = torch.Generator(device=m["device"] if m["device"] != "mps" else "cpu").manual_seed(seed + 99_999)
    return _fr_loop(m, seed, lambda state: nc.make_null_control_guidance_fn(generator=directions), settings)


def _amg_noise_level(step):
    """Noise level for lambda_t: 1 = pure noise, 0 = last step."""
    return 1.0 - step / max(N_STEPS - 1, 1)


def _amg_denoise(m, x, step, c3):
    """AMG-guided D(x) at sigmas[step] (Eqs. 10, 17, EDM form); NN search on decoded pixels (Supp. D)."""
    x_in = x.detach().requires_grad_(True)
    with torch.enable_grad():
        denoised = precond(m, x_in, m["sigmas"][step].expand(len(x_in)))
        pixels = m["vae"].decode(denoised / m["vae"].config.scaling_factor).sample
        shift, _, on = amg_paper.denoiser_guidance(x_in, pixels, training_pixels(m), _amg_noise_level(step), c3, m["sigmas"][step])
    return denoised.detach() + shift, bool(on.any())


def amg(m, seed, c3):
    """AMG with both denoiser evaluations of each Heun step guided."""
    x, active = z_init(m, seed), []
    for step in range(N_STEPS):
        guided, on = _amg_denoise(m, x, step, c3)
        active.append(on)
        with torch.no_grad():
            x = heun_step(m, x, step, guided, lambda z: _amg_denoise(m, z, step + 1, c3)[0])
    return _finish(x, active)


@torch.no_grad()
def _bm_proposal(m, x, step):
    """One Heun step and the proposal's x0_hat (which enters the proposal with coefficient 1 in EDM)."""
    proposal = heun_step(m, x, step, precond(m, x, m["sigmas"][step].expand(1)))
    x0_hat = precond(m, proposal, m["sigmas"][step + 1].expand(1)) if m["sigmas"][step + 1] > 0 else proposal
    return proposal, x0_hat


@torch.no_grad()
def norm_trace(m, seed):
    """Unguided endpoint and the three norms Broken Memories calibrates on, each [n_steps]."""
    x, trace = z_init(m, seed), {name: [] for name in bm.NORMS}
    for step in range(N_STEPS):
        proposal, x0_hat = _bm_proposal(m, x, step)
        for name, value in bm.norms(x, proposal, x0_hat).items():
            trace[name].append(value)
        x = proposal
    return x.cpu(), {name: torch.cat(v).cpu() for name, v in trace.items()}


@torch.no_grad()
def broken_memories(m, seed, stats, threshold_scale, mild_steps, gamma):
    mitigation = bm.Mitigation(stats, 1, m["device"], threshold_scale=threshold_scale, mild_steps=mild_steps, gamma=gamma)
    x, active = z_init(m, seed), []
    for step in range(N_STEPS):
        proposal, x0_hat = _bm_proposal(m, x, step)
        x, on = mitigation.step(step, x, proposal, x0_hat, 1.0)
        active.append(bool(on.any()))
    return _finish(x, active)
