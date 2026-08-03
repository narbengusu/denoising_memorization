"""Trains a low-rank JacNet: U such that M ~ U U^T approximates sigma_t^2 * dS/dz_t (the
rescaled score Jacobian), restricted by construction to its R dominant (highest-curvature,
"normal") directions -- see src/guidance/driver.py's make_lowrank_normal_projector_fn for how
this feeds guided sampling (project OUT the top-r' of these, rather than the dense-matrix
eig_tangent_projector's "keep the smallest-k eigenvalue directions").

Trains ONLY against the fine-tuned/memorized checkpoint (checkpoints/ddpm_finetuned.pt), not
the original pretrained one -- guidance here only ever queries JacNet around the memorized
model, so that's the only regime it needs to be accurate in.

Low-rank because a dense D x D output (cluster-codes/qm9/train_qm9_jacnet.py's approach) is
intractable at CIFAR-10's D=3072 (a dense output layer alone would need ~4.8B params at
hidden=512; materializing per-sample M as [B,D,D] would be gigabytes). The training objective
is otherwise the same denoising-Jacobian-matching relation as QM9's:
    resid = M_tilde + eps_hat eps_hat^T + I - eps eps^T,  M_tilde := U U^T
but ||resid||_F^2 is computed via an O(B*D*R) identity (see jac_loss below) that is
NUMERICALLY EXACT to the dense computation (verified against brute force on a small D before
this script was written) and never forms a D x D matrix.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # headless cluster node, no display -- save plots to file instead
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler

from cifar10_data import load_cifar10

config = {
    "device": os.environ.get("DEVICE", "cpu"),
    "pretrained_dir": os.environ.get("PRETRAINED_DIR", "pretrained/ddpm-cifar10-32"),
    "seed": 0,
    "jac_hidden": int(os.environ.get("JAC_HIDDEN", 512)),
    "jac_depth": int(os.environ.get("JAC_DEPTH", 4)),
    "jac_rank": int(os.environ.get("JAC_RANK", 8)),   # R -- JacNet's low-rank factor width
    "batch": int(os.environ.get("BATCH", 64)),
    "lr": float(os.environ.get("LR", 3e-4)),
    "iters": int(os.environ.get("ITERS", 50_000)),
    "log_every": int(os.environ.get("LOG_EVERY", 200)),
    "ckpt_every": int(os.environ.get("CKPT_EVERY", 2000)),
    "plot_every": int(os.environ.get("PLOT_EVERY", 2000)),
}
SCORE_CKPT_PATH = os.environ.get("SCORE_CKPT_PATH", "checkpoints/ddpm_finetuned.pt")   # fine-tuned/memorized only
TRAIN_IDX_PATH = os.environ.get("TRAIN_IDX_PATH", "checkpoints/train_idx.npy")
CKPT_PATH = os.environ.get("CKPT_PATH", "checkpoints/cifar10_jacnet.pt")
LOG_PATH = os.environ.get("LOG_PATH", "checkpoints/jacnet_train_log.txt")
PLOT_PATH = os.environ.get("PLOT_PATH", "checkpoints/jacnet_loss_curve.png")

torch.manual_seed(config["seed"])
os.makedirs("checkpoints", exist_ok=True)

D_FLAT = 3 * 32 * 32


def pack(x):
    return x.reshape(x.shape[0], -1)


images, _ = load_cifar10(split="train")
train_idx = np.load(TRAIN_IDX_PATH)
X_train = images[train_idx].to(config["device"])
print(f"training JacNet on the same {len(train_idx)}-image finetuning subset "
      f"({TRAIN_IDX_PATH})", flush=True)

scheduler = DDPMScheduler.from_pretrained(config["pretrained_dir"])
num_train_timesteps = scheduler.config.num_train_timesteps

score_net = UNet2DModel.from_pretrained(config["pretrained_dir"]).to(config["device"])
score_ckpt = torch.load(SCORE_CKPT_PATH, map_location=config["device"])
score_net.load_state_dict(score_ckpt.get("model_ema", score_ckpt["model"]))
score_net.eval()
for p in score_net.parameters():
    p.requires_grad_(False)
print(f"loaded FROZEN fine-tuned score net from {SCORE_CKPT_PATH} (using "
      f"{'EMA' if 'model_ema' in score_ckpt else 'raw'} weights, "
      f"trained {score_ckpt['iter'] + 1} iterations)", flush=True)


class JacNet(nn.Module):
    """Low-rank factor U: [B, D, R] such that M ~ U U^T -- see module docstring."""
    def __init__(self, d, hidden, depth, rank):
        super().__init__()
        self.d, self.rank = d, rank
        self.input_proj = nn.Linear(d + 1, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth * 2)])
        self.output_proj = nn.Linear(hidden, d * rank)

    def forward(self, z, t):
        h = F.silu(self.input_proj(torch.cat([z, t[:, None]], dim=-1)))
        for i in range(0, len(self.layers), 2):
            r = F.silu(self.layers[i](h))
            r = self.layers[i + 1](r)
            h = h + r
        return self.output_proj(h).reshape(z.shape[0], self.d, self.rank)


jac_net = JacNet(D_FLAT, config["jac_hidden"], config["jac_depth"], config["jac_rank"]).to(config["device"])
opt = torch.optim.Adam(jac_net.parameters(), lr=config["lr"])
print(f"JacNet parameters: {sum(p.numel() for p in jac_net.parameters()):,}  "
      f"(D_FLAT={D_FLAT}, rank={config['jac_rank']})", flush=True)


def jac_loss(x0):
    B = x0.shape[0]
    t_int = torch.randint(0, num_train_timesteps, (B,), device=config["device"]).long()
    t_frac = t_int.float() / (num_train_timesteps - 1)   # for the (1-t) loss weight below
    noise = torch.randn_like(x0)
    x_t = scheduler.add_noise(x0, noise, t_int)

    with torch.no_grad():
        eps_hat = score_net(x_t, t_int).sample

    z = pack(x_t)
    eps = pack(noise)
    eps_hat_flat = pack(eps_hat)

    U = jac_net(z, t_frac)   # [B, D_FLAT, R]

    # ||resid||_F^2 for resid = I + [U|eps_hat][U|eps_hat]^T - eps eps^T, computed via an
    # O(B*D*R) identity instead of ever forming a D_FLAT x D_FLAT matrix: expand
    # ||I + PP^T - nn^T||_F^2 using <I,PP^T>=trace(P^TP), ||PP^T||_F^2=||P^TP||_F^2,
    # <PP^T,nn^T>=||P^Tn||^2, where P=[U|eps_hat], n=eps -- verified exact against a dense
    # brute-force computation on a small D before this script was written.
    Pmat = torch.cat([U, eps_hat_flat.unsqueeze(-1)], dim=-1)     # [B, D_FLAT, R+1]
    G = torch.bmm(Pmat.transpose(-1, -2), Pmat)                   # [B, R+1, R+1]
    Pn = torch.bmm(Pmat.transpose(-1, -2), eps.unsqueeze(-1)).squeeze(-1)   # [B, R+1]
    eps_sqnorm = (eps * eps).sum(-1)                              # ||eps||^2, [B]

    frob2 = (D_FLAT + (G ** 2).sum(dim=(1, 2)) + eps_sqnorm ** 2
             + 2 * G.diagonal(dim1=1, dim2=2).sum(-1) - 2 * eps_sqnorm - 2 * (Pn ** 2).sum(-1))

    w_t = (1.0 - t_frac).clamp(min=0)
    loss = (w_t * frob2 / D_FLAT ** 2).mean()

    # Diagnostic only (does not affect the gradient/training objective above): the
    # diagonal-only residual (row_sumsq(U)_i + eps_hat_i^2 + 1 - eps_i^2) is the part
    # score_div_guidance's I(y)=trace(M)+.. actually consumes -- mirrors QM9's
    # diag_mse/offdiag_mse split (see train_qm9_jacnet.py's jac_loss comment).
    with torch.no_grad():
        diag_resid = (U ** 2).sum(-1) + eps_hat_flat ** 2 + 1.0 - eps ** 2   # [B, D_FLAT]
        diag_mse = (w_t * (diag_resid ** 2).sum(-1) / D_FLAT).mean()
        offdiag_ss = frob2 - (diag_resid ** 2).sum(-1)
        offdiag_mse = (w_t * offdiag_ss / (D_FLAT ** 2 - D_FLAT)).mean()

    return loss, diag_mse.item(), offdiag_mse.item()


start_iter = 0
log_iters, train_hist, diag_hist, offdiag_hist = [], [], [], []
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=config["device"])
    jac_net.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    start_iter = ckpt["iter"] + 1
    log_iters, train_hist = ckpt.get("log_iters", []), ckpt.get("train_hist", [])
    diag_hist, offdiag_hist = ckpt.get("diag_hist", []), ckpt.get("offdiag_hist", [])
    print(f"resumed from checkpoint at iter {start_iter}", flush=True)


def save_loss_plot():
    if not log_iters:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(log_iters, train_hist, color="steelblue")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="iteration", ylabel="loss (log scale)", title="JacNet full loss (low-rank residual)")
    axes[1].plot(log_iters, diag_hist, color="tomato", label="diagonal MSE (what I(y) uses)")
    axes[1].plot(log_iters, offdiag_hist, color="steelblue", alpha=0.6, label="off-diagonal MSE")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="iteration", ylabel="MSE (log scale)", title="diagonal vs. off-diagonal")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    plt.close()


log_f = open(LOG_PATH, "a")
rng = np.random.default_rng(config["seed"])
t0 = time.time()
for i in range(start_iter, config["iters"]):
    idx = rng.integers(0, len(train_idx), config["batch"])
    loss, diag_mse, offdiag_mse = jac_loss(X_train[idx])
    opt.zero_grad(); loss.backward(); opt.step()

    if i % config["log_every"] == 0:
        elapsed = time.time() - t0
        msg = (f"iter {i:7d}  loss={loss.item():.4f}  diag_mse={diag_mse:.4f}  "
               f"offdiag_mse={offdiag_mse:.4f}  elapsed={elapsed:.0f}s")
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()
        log_iters.append(i); train_hist.append(loss.item())
        diag_hist.append(diag_mse); offdiag_hist.append(offdiag_mse)

    if i % config["ckpt_every"] == 0 and i > start_iter:
        torch.save({"model": jac_net.state_dict(), "opt": opt.state_dict(), "iter": i,
                    "config": config, "score_ckpt_path": SCORE_CKPT_PATH,
                    "log_iters": log_iters, "train_hist": train_hist,
                    "diag_hist": diag_hist, "offdiag_hist": offdiag_hist},
                   CKPT_PATH)

    if i % config["plot_every"] == 0 and i > start_iter:
        save_loss_plot()

torch.save({"model": jac_net.state_dict(), "opt": opt.state_dict(), "iter": config["iters"] - 1,
            "config": config, "score_ckpt_path": SCORE_CKPT_PATH,
            "log_iters": log_iters, "train_hist": train_hist,
            "diag_hist": diag_hist, "offdiag_hist": offdiag_hist},
           CKPT_PATH)
save_loss_plot()
print("training complete", flush=True)
