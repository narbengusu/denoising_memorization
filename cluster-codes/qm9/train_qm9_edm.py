import os
import time
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless cluster node, no display -- save plots to file instead
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from rdkit import Chem

config = {
    "sdf_path": "data/gdb9/gdb9.sdf",
    "device": os.environ.get("DEVICE", "cpu"),
    "elements": ["H", "C", "N", "O", "F"],
    "h_scale": 0.25,
    "T": 1.0,
    "diffusion_steps": int(os.environ.get("DIFFUSION_STEPS", 500)),  # EDM's QM9 default
    "poly_power": float(os.environ.get("POLY_POWER", 2.0)),          # EDM's QM9 default ("polynomial_2")
    "poly_s": float(os.environ.get("POLY_S", 1e-4)),
    "poly_clip": float(os.environ.get("POLY_CLIP", 0.001)),
    "hidden": int(os.environ.get("HIDDEN", 256)),      # matches EDM paper (was 128 in the first run)
    "n_layers": int(os.environ.get("N_LAYERS", 9)),    # matches EDM paper (was 4 in the first run)
    "batch": 64, "lr": 1e-4,
    "ema_decay": float(os.environ.get("EMA_DECAY", 0.999)),
    "seed": 0,
    "iters": int(os.environ.get("ITERS", 200_000)),
    "n_train": int(os.environ.get("N_TRAIN", 100_000)),
    "n_val": int(os.environ.get("N_VAL", 18_000)),
    "log_every": int(os.environ.get("LOG_EVERY", 200)),
    "ckpt_every": int(os.environ.get("CKPT_EVERY", 2000)),
    "plot_every": int(os.environ.get("PLOT_EVERY", 2000)),
    "ckpt_path": "checkpoints/qm9_edm.pt",
    "log_path": "checkpoints/train_log.txt",
    "plot_path": "checkpoints/loss_curve.png",
}
torch.manual_seed(config["seed"])
os.makedirs("checkpoints", exist_ok=True)


def hill_formula(symbols):
    c = Counter(symbols)
    order = [e for e in ("C", "H") if e in c] + sorted(e for e in c if e not in ("C", "H"))
    return "".join(f"{e}{c[e] if c[e] > 1 else ''}" for e in order)


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
    # Exact port of EDM's polynomial_schedule + clip_noise_schedule (en_diffusion.py). alphas2[t_int]
    # is what we call alpha_bar(t) -- the closed-form signal-retention fraction at discretized step t_int.
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
    # t: continuous in [0, T] -> nearest of (diffusion_steps + 1) precomputed polynomial-schedule bins
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


def loss_fn(net, x0, h0, node_mask):
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device).clamp(min=1e-3) * config["T"]
    xt, ht, eps_x, eps_h = q_sample(x0, h0, t, node_mask)
    eps_x_hat, eps_h_hat = net(xt, ht, t, node_mask)
    n_real = node_mask.sum(dim=1).clamp(min=1.0)
    loss_x = ((eps_x_hat - eps_x) ** 2).sum(dim=(1, 2)) / n_real
    loss_h = ((eps_h_hat - eps_h) ** 2).sum(dim=(1, 2)) / n_real
    return (loss_x + loss_h).mean()


net = EGNN(config["n_types"], config["hidden"], config["n_layers"]).to(config["device"])
opt = torch.optim.Adam(net.parameters(), lr=config["lr"])
print(f"EGNN parameters: {sum(p.numel() for p in net.parameters()):,}", flush=True)

# EMA of the weights -- smooths out the noisy tail of SGD, standard practice in diffusion
# training (used in EDM itself). Sampling should use ema_state, not the raw net weights.
ema_state = {k: v.clone().detach() for k, v in net.state_dict().items()}


def update_ema():
    with torch.no_grad():
        for k, v in net.state_dict().items():
            ema_state[k].mul_(config["ema_decay"]).add_(v, alpha=1 - config["ema_decay"])


start_iter = 0
log_iters, train_hist, val_hist = [], [], []
if os.path.exists(config["ckpt_path"]):
    ckpt = torch.load(config["ckpt_path"], map_location=config["device"])
    net.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    ema_state = ckpt.get("model_ema", ema_state)
    start_iter = ckpt["iter"] + 1
    log_iters, train_hist, val_hist = ckpt.get("log_iters", []), ckpt.get("train_hist", []), ckpt.get("val_hist", [])
    print(f"resumed from checkpoint at iter {start_iter}", flush=True)


def save_loss_plot():
    if not log_iters:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(log_iters, train_hist, label="train", color="steelblue")
    plt.plot(log_iters, val_hist, label="val", color="firebrick")
    plt.yscale("log")
    plt.xlabel("iteration"); plt.ylabel("loss (log scale)")
    plt.legend(); plt.title(f"QM9 EDM training loss (iters={log_iters[-1]})")
    plt.tight_layout()
    plt.savefig(config["plot_path"], dpi=120)
    plt.close()


log_f = open(config["log_path"], "a")
t0 = time.time()
for i in range(start_iter, config["iters"]):
    idx = torch.randint(0, N_TRAIN, (config["batch"],))
    loss = loss_fn(net, X_train[idx], H_train[idx], M_train[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    update_ema()

    if i % config["log_every"] == 0:
        with torch.no_grad():
            vidx = torch.randint(0, N_VAL, (config["batch"],))
            val_loss = loss_fn(net, X_val[vidx], H_val[vidx], M_val[vidx]).item()
        elapsed = time.time() - t0
        msg = f"iter {i:7d}  loss={loss.item():.4f}  val_loss={val_loss:.4f}  elapsed={elapsed:.0f}s"
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()
        log_iters.append(i); train_hist.append(loss.item()); val_hist.append(val_loss)

    if i % config["ckpt_every"] == 0 and i > start_iter:
        torch.save({"model": net.state_dict(), "model_ema": ema_state, "opt": opt.state_dict(), "iter": i,
                    "config": config, "log_iters": log_iters, "train_hist": train_hist, "val_hist": val_hist},
                   config["ckpt_path"])

    if i % config["plot_every"] == 0 and i > start_iter:
        save_loss_plot()

torch.save({"model": net.state_dict(), "model_ema": ema_state, "opt": opt.state_dict(), "iter": config["iters"] - 1,
            "config": config, "log_iters": log_iters, "train_hist": train_hist, "val_hist": val_hist},
           config["ckpt_path"])
save_loss_plot()
print("training complete", flush=True)
