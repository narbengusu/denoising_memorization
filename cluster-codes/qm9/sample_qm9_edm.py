import os
import time
import numpy as np
import torch
import torch.nn as nn
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
    "hidden": int(os.environ.get("HIDDEN", 256)),
    "n_layers": int(os.environ.get("N_LAYERS", 9)),
    "seed": 0,
    "n_train": int(os.environ.get("N_TRAIN", 100_000)),
    "n_val": int(os.environ.get("N_VAL", 18_000)),
}
CKPT_PATH = os.environ.get("CKPT_PATH", "checkpoints/qm9_edm.pt")
OUT_PATH = os.environ.get("OUT_PATH", "checkpoints/samples.pt")
N_EVAL = int(os.environ.get("N_EVAL", 200))
N_STEPS_EVAL = int(os.environ.get("N_STEPS_EVAL", 500))
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", 1))

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
train_idx = perm[:N_TRAIN]
test_idx_all = perm[N_TRAIN + N_VAL:]

X_test = torch.from_numpy(X_all[test_idx_all]).to(config["device"])
H_test = torch.from_numpy(H_all[test_idx_all]).to(config["device"])
M_test = torch.from_numpy(mask_all[test_idx_all]).to(config["device"])

atom_count_probs = np.bincount(mask_all.sum(1).astype(int), minlength=MAX_ATOMS + 1) / N_mol


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


@torch.no_grad()
def sample_node_masks(B, rng_np):
    counts = rng_np.choice(MAX_ATOMS + 1, size=B, p=atom_count_probs)
    node_mask = torch.zeros(B, MAX_ATOMS)
    for i, c in enumerate(counts):
        node_mask[i, :c] = 1.0
    return node_mask.to(config["device"])


@torch.no_grad()
def sample(net, B, n_steps=500, node_mask=None, rng_np=None):
    # Exact discrete-time ancestral posterior (matches EDM's sample_p_zs_given_zt), not an
    # SDE Euler-Maruyama approximation -- see qm9_fisher_rao.ipynb for the derivation notes.
    rng_np = rng_np or np.random.default_rng()
    node_mask = node_mask if node_mask is not None else sample_node_masks(B, rng_np)
    x = com_project(torch.randn(B, MAX_ATOMS, 3, device=config["device"]), node_mask)
    h = torch.randn(B, MAX_ATOMS, config["n_types"], device=config["device"]) * node_mask.unsqueeze(-1)
    dt = config["T"] / n_steps

    for k in reversed(range(n_steps)):
        t_vec = torch.full((B,), (k + 1) * dt, device=config["device"])
        s_vec = torch.full((B,), k * dt, device=config["device"])

        eps_x_hat, eps_h_hat = net(x, h, t_vec, node_mask)

        a_t = alpha_bar(t_vec)[:, None, None]
        a_s = alpha_bar(s_vec)[:, None, None]
        sigma_t = (1 - a_t).sqrt()
        sigma_s = (1 - a_s).sqrt()
        alpha_t_given_s = (a_t / a_s).sqrt()
        sigma2_t_given_s = 1 - a_t / a_s
        post_sigma = sigma2_t_given_s.sqrt() * sigma_s / sigma_t

        mu_x = x / alpha_t_given_s - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)) * eps_x_hat
        mu_h = h / alpha_t_given_s - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)) * eps_h_hat

        if k > 0:
            noise_x = com_project(torch.randn_like(x), node_mask)
            noise_h = torch.randn_like(h) * node_mask.unsqueeze(-1)
        else:
            noise_x = noise_h = 0

        x = com_project(mu_x + post_sigma * noise_x, node_mask)
        h = mu_h + post_sigma * noise_h

    types = (h / config["h_scale"]).argmax(dim=-1)
    return x.cpu(), types.cpu(), node_mask.cpu()


BOND_LEN_1 = {
    ("H", "H"): 0.74, ("H", "C"): 1.09, ("H", "N"): 1.01, ("H", "O"): 0.96, ("H", "F"): 0.92,
    ("C", "C"): 1.54, ("C", "N"): 1.47, ("C", "O"): 1.43, ("C", "F"): 1.35,
    ("N", "N"): 1.45, ("N", "O"): 1.40, ("N", "F"): 1.36,
    ("O", "O"): 1.48, ("O", "F"): 1.42,
    ("F", "F"): 1.42,
}
BOND_LEN_2 = {
    ("C", "C"): 1.34, ("C", "N"): 1.29, ("C", "O"): 1.20,
    ("N", "N"): 1.25, ("N", "O"): 1.21, ("O", "O"): 1.21,
}
BOND_LEN_3 = {("C", "C"): 1.20, ("C", "N"): 1.16, ("N", "N"): 1.10}
MARGIN_1, MARGIN_2, MARGIN_3 = 0.10, 0.05, 0.03
ALLOWED_VALENCE = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}


def _lookup(table, a, b):
    return table.get((a, b), table.get((b, a)))


def bond_order(sym_a, sym_b, dist):
    d3 = _lookup(BOND_LEN_3, sym_a, sym_b)
    if d3 is not None and dist < d3 + MARGIN_3:
        return 3
    d2 = _lookup(BOND_LEN_2, sym_a, sym_b)
    if d2 is not None and dist < d2 + MARGIN_2:
        return 2
    d1 = _lookup(BOND_LEN_1, sym_a, sym_b)
    if d1 is not None and dist < d1 + MARGIN_1:
        return 1
    return 0


def molecule_stability(positions, symbols):
    n = len(symbols)
    valence = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            order = bond_order(symbols[i], symbols[j], dist)
            if order > 0:
                valence[i] += order
                valence[j] += order
    atom_stable = np.array([valence[i] == ALLOWED_VALENCE[symbols[i]] for i in range(n)])
    return atom_stable, bool(atom_stable.all())


def evaluate_stability(X, types, mask):
    n_atoms_stable, n_atoms_total, n_mol_stable = 0, 0, 0
    B = X.shape[0]
    for b in range(B):
        n = int(mask[b].sum().item())
        pos = X[b, :n].numpy()
        syms = [config["elements"][t] for t in types[b, :n].tolist()]
        atom_stable, mol_stable = molecule_stability(pos, syms)
        n_atoms_stable += atom_stable.sum()
        n_atoms_total += n
        n_mol_stable += int(mol_stable)
    return {"atom_stability": float(n_atoms_stable) / max(n_atoms_total, 1), "molecule_stability": float(n_mol_stable) / B}


ckpt = torch.load(CKPT_PATH, map_location=config["device"])
assert ckpt["config"]["hidden"] == config["hidden"] and ckpt["config"]["n_layers"] == config["n_layers"], \
    "checkpoint architecture doesn't match this script's config[\"hidden\"]/[\"n_layers\"]"

net = EGNN(config["n_types"], config["hidden"], config["n_layers"]).to(config["device"])
state_dict = ckpt.get("model_ema", ckpt["model"])
net.load_state_dict(state_dict)
net.eval()
print(f"loaded checkpoint from {CKPT_PATH}  (using {'EMA' if 'model_ema' in ckpt else 'raw'} weights, "
      f"trained {ckpt['iter'] + 1} iterations)", flush=True)

t0 = time.time()
x_gen, types_gen, mask_gen = sample(net, N_EVAL, n_steps=N_STEPS_EVAL, rng_np=np.random.default_rng(SAMPLE_SEED))
print(f"generated {N_EVAL} molecules with n_steps={N_STEPS_EVAL} in {time.time() - t0:.1f}s", flush=True)

gen_stats = evaluate_stability(x_gen, types_gen, mask_gen)
print("generated-sample stability:", gen_stats, flush=True)

test_idx = np.random.default_rng(SAMPLE_SEED).choice(len(X_test), size=N_EVAL, replace=False)
test_types = H_test[test_idx].argmax(dim=-1)
test_stats = evaluate_stability(X_test[test_idx].cpu(), test_types.cpu(), M_test[test_idx].cpu())
print("held-out real test-set stability:", test_stats, flush=True)

torch.save({
    "x_gen": x_gen, "types_gen": types_gen, "mask_gen": mask_gen,
    "gen_stats": gen_stats, "test_stats": test_stats,
    "test_idx": torch.from_numpy(test_idx), "n_eval": N_EVAL, "n_steps_eval": N_STEPS_EVAL,
    "ckpt_iter": int(ckpt["iter"]), "ckpt_path": CKPT_PATH,
}, OUT_PATH)
print(f"saved samples + stats to {OUT_PATH}", flush=True)
