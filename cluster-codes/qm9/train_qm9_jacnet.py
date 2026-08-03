import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless cluster node, no display -- save plots to file instead
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

config = {
    "sdf_path": "data/gdb9/gdb9.sdf",
    "device": os.environ.get("DEVICE", "cpu"),
    "elements": ["H", "C", "N", "O", "F"],
    "h_scale": 0.25,
    "T": 1.0,
    "diffusion_steps": int(os.environ.get("DIFFUSION_STEPS", 500)),
    "poly_power": float(os.environ.get("POLY_POWER", 2.0)),
    "poly_s": float(os.environ.get("POLY_S", 1e-4)),
    "poly_clip": float(os.environ.get("POLY_CLIP", 0.001)),
    "hidden": int(os.environ.get("HIDDEN", 256)),      # score net architecture -- MUST match SCORE_CKPT_PATH
    "n_layers": int(os.environ.get("N_LAYERS", 9)),
    "jac_hidden": int(os.environ.get("JAC_HIDDEN", 512)),
    "jac_depth": int(os.environ.get("JAC_DEPTH", 4)),
    "batch": int(os.environ.get("BATCH", 128)),
    "lr": float(os.environ.get("LR", 3e-4)),
    "seed": 0,
    "iters": int(os.environ.get("ITERS", 200_000)),
    "n_train": int(os.environ.get("N_TRAIN", 100_000)),
    "n_val": int(os.environ.get("N_VAL", 18_000)),
    "log_every": int(os.environ.get("LOG_EVERY", 200)),
    "ckpt_every": int(os.environ.get("CKPT_EVERY", 2000)),
    "plot_every": int(os.environ.get("PLOT_EVERY", 2000)),
}
SCORE_CKPT_PATH = os.environ.get("SCORE_CKPT_PATH", "checkpoints/qm9_edm.pt")
CKPT_PATH = os.environ.get("CKPT_PATH", "checkpoints/qm9_jacnet.pt")
LOG_PATH = os.environ.get("LOG_PATH", "checkpoints/jacnet_train_log.txt")
PLOT_PATH = os.environ.get("PLOT_PATH", "checkpoints/jacnet_loss_curve.png")

torch.manual_seed(config["seed"])
os.makedirs("checkpoints", exist_ok=True)


print("parsing SDF...", flush=True)
t0 = time.time()
supplier = Chem.SDMolSupplier(config["sdf_path"], removeHs=False, sanitize=False)
records = []
for mol in supplier:
    if mol is None:
        continue
    conf = mol.GetConformer()
    symbols = tuple(a.GetSymbol() for a in mol.GetAtoms())
    records.append({
        "num_atoms": mol.GetNumAtoms(),
        "symbols": symbols,
        "positions": conf.GetPositions().astype(np.float32),
    })
print(f"parsed {len(records)} molecules in {time.time() - t0:.1f}s", flush=True)

ELEMENT_TO_IDX = {e: i for i, e in enumerate(config["elements"])}
MAX_ATOMS = max(r["num_atoms"] for r in records)
config["max_atoms"] = MAX_ATOMS
config["n_types"] = len(config["elements"])

N_mol = len(records)
X_all = np.zeros((N_mol, MAX_ATOMS, 3), dtype=np.float32)
H_all = np.zeros((N_mol, MAX_ATOMS, config["n_types"]), dtype=np.float32)
mask_all = np.zeros((N_mol, MAX_ATOMS), dtype=np.float32)
for i, r in enumerate(records):
    n = r["num_atoms"]
    X_all[i, :n] = r["positions"]
    mask_all[i, :n] = 1.0
    for j, s in enumerate(r["symbols"]):
        H_all[i, j, ELEMENT_TO_IDX[s]] = 1.0

centroid = (X_all * mask_all[..., None]).sum(axis=1, keepdims=True) / mask_all.sum(axis=1, keepdims=True)[..., None]
X_all = (X_all - centroid) * mask_all[..., None]

rng = np.random.default_rng(config["seed"])
perm = rng.permutation(N_mol)
N_TRAIN, N_VAL = config["n_train"], config["n_val"]
train_idx, val_idx = perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + N_VAL]

X_train = torch.from_numpy(X_all[train_idx]).to(config["device"])
H_train = torch.from_numpy(H_all[train_idx]).to(config["device"])
M_train = torch.from_numpy(mask_all[train_idx]).to(config["device"])
X_val = torch.from_numpy(X_all[val_idx]).to(config["device"])
H_val = torch.from_numpy(H_all[val_idx]).to(config["device"])
M_val = torch.from_numpy(mask_all[val_idx]).to(config["device"])
print(f"train={N_TRAIN} val={N_VAL}", flush=True)


# ---------------------------------------------------------------------------
# Frozen score network (exact copy of the EGNN in train_qm9_edm.py / sample_qm9_edm.py --
# architecture must match SCORE_CKPT_PATH exactly, loaded read-only below).
# ---------------------------------------------------------------------------
class EGNNLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h, x, edge_mask):
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        dist = diff.norm(dim=-1, keepdim=True)
        dist2 = dist ** 2
        N = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        m_ij = self.edge_mlp(torch.cat([h_i, h_j, dist2], dim=-1))
        m_ij = m_ij * edge_mask.unsqueeze(-1)
        coord_w = self.coord_mlp(m_ij)
        diff_norm = diff / (dist + 1.0)
        n_nbr = edge_mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        x_update = (diff_norm * coord_w * edge_mask.unsqueeze(-1)).sum(dim=2) / n_nbr
        x = x + x_update
        m_i = m_ij.sum(dim=2)
        h = h + self.node_mlp(torch.cat([h, m_i], dim=-1))
        return h, x


class EGNN(nn.Module):
    def __init__(self, n_types, hidden, n_layers):
        super().__init__()
        self.embed = nn.Linear(n_types + 1, hidden)
        self.layers = nn.ModuleList([EGNNLayer(hidden) for _ in range(n_layers)])
        self.h_out = nn.Linear(hidden, n_types)

    def forward(self, x, h, t, node_mask):
        B, N, _ = x.shape
        t_feat = t[:, None, None].expand(-1, N, 1)
        h0 = self.embed(torch.cat([h, t_feat], dim=-1)) * node_mask.unsqueeze(-1)
        edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        edge_mask = edge_mask * (1 - torch.eye(N, device=x.device))[None]
        h_l, x_l = h0, x
        for layer in self.layers:
            h_l, x_l = layer(h_l, x_l, edge_mask)
            h_l = h_l * node_mask.unsqueeze(-1)
            x_l = x_l * node_mask.unsqueeze(-1)
        eps_x = (x_l - x) * node_mask.unsqueeze(-1)
        eps_h = self.h_out(h_l) * node_mask.unsqueeze(-1)
        return eps_x, eps_h


def _build_polynomial_alpha_bar_table(timesteps, s, power, clip_value):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - (x / steps) ** power) ** 2
    padded = np.concatenate([np.ones(1), alphas2])
    alphas_step = np.clip(padded[1:] / padded[:-1], clip_value, 1.0)
    alphas2 = np.cumprod(alphas_step)
    precision = 1 - 2 * s
    return precision * alphas2 + s


_ALPHA_BAR_TABLE = torch.from_numpy(_build_polynomial_alpha_bar_table(
    config["diffusion_steps"], config["poly_s"], config["poly_power"], config["poly_clip"]
)).float().to(config["device"])


def alpha_bar(t):
    t_int = torch.round(t / config["T"] * config["diffusion_steps"]).long().clamp(0, config["diffusion_steps"])
    return _ALPHA_BAR_TABLE.to(t.device)[t_int]


def com_project(v, node_mask):
    n_real = node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (v * node_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / n_real.unsqueeze(-1)
    return (v - mean) * node_mask.unsqueeze(-1)


def q_sample(x0, h0, t, node_mask):
    eps_x = com_project(torch.randn_like(x0), node_mask)
    eps_h = torch.randn_like(h0) * node_mask.unsqueeze(-1)
    a = alpha_bar(t)[:, None, None]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps_x
    ht = a.sqrt() * (h0 * config["h_scale"]) + (1 - a).sqrt() * eps_h
    return xt, ht, eps_x, eps_h


score_ckpt = torch.load(SCORE_CKPT_PATH, map_location=config["device"])
assert score_ckpt["config"]["hidden"] == config["hidden"] and score_ckpt["config"]["n_layers"] == config["n_layers"], \
    "checkpoint architecture doesn't match this script's config[\"hidden\"]/[\"n_layers\"]"
score_net = EGNN(config["n_types"], config["hidden"], config["n_layers"]).to(config["device"])
score_net.load_state_dict(score_ckpt.get("model_ema", score_ckpt["model"]))
score_net.eval()
for p in score_net.parameters():
    p.requires_grad_(False)
print(f"loaded FROZEN score net from {SCORE_CKPT_PATH} "
      f"(using {'EMA' if 'model_ema' in score_ckpt else 'raw'} weights, "
      f"trained {score_ckpt['iter'] + 1} iterations)", flush=True)


# ---------------------------------------------------------------------------
# Flat-vector packing (same convention as fr_guidance.py's sample scripts) --
# JacNet operates on the padded flat [B, D] state, D = MAX_ATOMS*3 + MAX_ATOMS*n_types.
# ---------------------------------------------------------------------------
D_FLAT = MAX_ATOMS * 3 + MAX_ATOMS * config["n_types"]


def pack(x, h):
    return torch.cat([x.reshape(x.shape[0], -1), h.reshape(h.shape[0], -1)], dim=-1)


def unpack(z):
    B = z.shape[0]
    x_flat, h_flat = z[:, :MAX_ATOMS * 3], z[:, MAX_ATOMS * 3:]
    return x_flat.reshape(B, MAX_ATOMS, 3), h_flat.reshape(B, MAX_ATOMS, config["n_types"])


def state_mask(node_mask):
    # [B, MAX_ATOMS] node mask -> [B, D_FLAT] per-scalar mask (1 for a real atom's 3
    # coordinate / n_types channel entries, 0 for a padded atom's).
    B = node_mask.shape[0]
    mx = node_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(B, -1)
    mh = node_mask.unsqueeze(-1).expand(-1, -1, config["n_types"]).reshape(B, -1)
    return torch.cat([mx, mh], dim=-1)


# ---------------------------------------------------------------------------
# JacNet: M_psi(z_t, t) ~ sigma_t^2 * dS/dz_t (rescaled score Jacobian), trained with a
# denoising-Jacobian-matching loss -- flat residual MLP, same architecture family as
# jacobian_groundtruth_experiment.ipynb's JacNet, operating on the packed flat state.
# ---------------------------------------------------------------------------
class JacNet(nn.Module):
    def __init__(self, d, hidden, depth):
        super().__init__()
        self.input_proj = nn.Linear(d + 1, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth * 2)])
        self.output_proj = nn.Linear(hidden, d * d)

    def forward(self, z, t):
        h = F.silu(self.input_proj(torch.cat([z, t[:, None]], dim=-1)))
        for i in range(0, len(self.layers), 2):
            r = F.silu(self.layers[i](h))
            r = self.layers[i + 1](r)
            h = h + r
        return self.output_proj(h)


jac_net = JacNet(D_FLAT, config["jac_hidden"], config["jac_depth"]).to(config["device"])
opt = torch.optim.Adam(jac_net.parameters(), lr=config["lr"])
print(f"JacNet parameters: {sum(p.numel() for p in jac_net.parameters()):,}  (D_FLAT={D_FLAT})", flush=True)

_ID_FLAT = torch.eye(D_FLAT, device=config["device"]).unsqueeze(0)


def jac_net_symmetric(z, t):
    # Symmetrize the raw output: the true target (eps eps^T terms, I) is exactly
    # symmetric, but jac_net's raw dense output has no such constraint, forcing it to
    # learn symmetry from scratch across ~2x the effectively-independent outputs it
    # needs to. Also required downstream for eigendecomposition -- torch.linalg.eigh
    # (real eigenvalues, orthogonal eigenvectors, numerically stable) needs a genuinely
    # symmetric input; an asymmetric M would need the slower/noisier torch.linalg.eig.
    M_raw = jac_net(z, t).reshape(z.shape[0], D_FLAT, D_FLAT)
    return 0.5 * (M_raw + M_raw.transpose(-1, -2))


def jac_loss(x0, h0, node_mask):
    # M_tilde(x_t,t) = sigma_t^2 * M ~ sigma_t^2 * dS/dz_t. Denoising-Jacobian-matching
    # objective (Tweedie-based, general -- not GMM-specific, see
    # jacobian_groundtruth_experiment.ipynb for the derivation and a closed-form check):
    #   resid = M_tilde + eps_hat eps_hat^T + I - eps eps^T,  weighted by w(t) = (1-t)
    # Padded-atom entries are masked out of BOTH M_tilde and the target terms -- the
    # unmasked formula would otherwise force M_tilde to learn -1 on padded diagonal
    # entries (since eps/eps_hat are exactly zero there but the identity term isn't),
    # which doesn't match the autograd ground truth (the frozen score net's own output,
    # and hence its Jacobian, is forced to exactly zero at padded dims by node_mask
    # multiplication -- see EGNN.forward).
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device).clamp(min=1e-3) * config["T"]
    xt, ht, eps_x, eps_h = q_sample(x0, h0, t, node_mask)
    with torch.no_grad():
        eps_x_hat, eps_h_hat = score_net(xt, ht, t, node_mask)

    z = pack(xt, ht)
    eps = pack(eps_x, eps_h)
    eps_hat = pack(eps_x_hat, eps_h_hat)
    smask = state_mask(node_mask)
    mask2 = smask.unsqueeze(2) * smask.unsqueeze(1)

    M_tilde = jac_net_symmetric(z, t) * mask2
    eeT = torch.bmm(eps.unsqueeze(2), eps.unsqueeze(1)) * mask2
    eehT = torch.bmm(eps_hat.unsqueeze(2), eps_hat.unsqueeze(1)) * mask2
    Id = _ID_FLAT * mask2

    resid = M_tilde + eehT + Id - eeT
    w_t = (1.0 - t).clamp(min=0)
    n_real_dims = smask.sum(dim=1).clamp(min=1.0)   # normalize by the sample's own real dim count
    loss = (w_t * (resid ** 2).sum(dim=(1, 2)) / n_real_dims ** 2).mean()

    # Diagnostic only (does not affect the gradient/training objective above): of the
    # D_FLAT^2 = 53,824 residual entries, only D_FLAT = 232 are diagonal -- the ones
    # score_div_guidance's I(y) = trace(M) + d/sigma_t^2 actually consumes. The other
    # 99.6% are cross terms eps_i*eps_j (i != j), which are near-unlearnable single-sample
    # noise around ~0 (eps is drawn once per example) -- they can swamp the aggregate loss
    # above and hide whether the diagonal (the part that matters) is improving at all.
    with torch.no_grad():
        diag_resid = resid.diagonal(dim1=1, dim2=2)                              # [B, D_FLAT]
        diag_mse = (w_t * (diag_resid ** 2).sum(-1) / n_real_dims).mean()
        offdiag_ss = (resid ** 2).sum(dim=(1, 2)) - (diag_resid ** 2).sum(-1)
        n_offdiag = (n_real_dims ** 2 - n_real_dims).clamp(min=1.0)
        offdiag_mse = (w_t * offdiag_ss / n_offdiag).mean()

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
    axes[0].set(xlabel="iteration", ylabel="loss (log scale)", title="JacNet full loss (all D^2 entries)")
    axes[1].plot(log_iters, diag_hist, color="tomato", label="diagonal MSE (what I(y) uses)")
    axes[1].plot(log_iters, offdiag_hist, color="steelblue", alpha=0.6, label="off-diagonal MSE")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="iteration", ylabel="MSE (log scale)", title="diagonal vs. off-diagonal")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    plt.close()


log_f = open(LOG_PATH, "a")
t0 = time.time()
for i in range(start_iter, config["iters"]):
    idx = torch.randint(0, N_TRAIN, (config["batch"],))
    loss, diag_mse, offdiag_mse = jac_loss(X_train[idx], H_train[idx], M_train[idx])
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
