import os
import time
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem

config = {
    "sdf_path": "data/gdb9/gdb9.sdf",
    "elements": ["H", "C", "N", "O", "F"],
    "seed": 0,
    "n_train": int(os.environ.get("N_TRAIN", 100_000)),
    "n_val": int(os.environ.get("N_VAL", 18_000)),
}
SAMPLES_PATH = os.environ.get("SAMPLES_PATH", "checkpoints/samples.pt")
OUT_PATH = os.environ.get("OUT_PATH", "checkpoints/memorization_results.pt")
PLOT_PATH = os.environ.get("PLOT_PATH", "checkpoints/memorization_plot.png")
MAX_CANDIDATES_PER_BUCKET = int(os.environ.get("MAX_CANDIDATES_PER_BUCKET", 1000))
N_RESTARTS = int(os.environ.get("N_RESTARTS", 100))
N_WORKERS = int(os.environ.get("N_WORKERS", os.cpu_count()))


# ---------------------------------------------------------------------------
# Data: parse SDF, build padded arrays, recreate the exact same train/val/test
# split (same seed) used everywhere else in this project. Wrapped in a function
# (called only from the `if __name__ == "__main__":` guard below) so that
# spawned worker processes -- which re-import this module but don't need any
# of this data, only `_align_worker` -- don't waste time re-parsing the SDF.
# ---------------------------------------------------------------------------
def load_data():
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

    element_to_idx = {e: i for i, e in enumerate(config["elements"])}
    max_atoms = max(r["num_atoms"] for r in records)
    n_mol = len(records)
    x_all = np.zeros((n_mol, max_atoms, 3), dtype=np.float32)
    h_all = np.zeros((n_mol, max_atoms, len(config["elements"])), dtype=np.float32)
    mask_all = np.zeros((n_mol, max_atoms), dtype=np.float32)
    for i, r in enumerate(records):
        n = r["num_atoms"]
        x_all[i, :n] = r["positions"]
        mask_all[i, :n] = 1.0
        for j, s in enumerate(r["symbols"]):
            h_all[i, j, element_to_idx[s]] = 1.0
    centroid = (x_all * mask_all[..., None]).sum(axis=1, keepdims=True) / mask_all.sum(axis=1, keepdims=True)[..., None]
    x_all = (x_all - centroid) * mask_all[..., None]

    rng_split = np.random.default_rng(config["seed"])
    perm = rng_split.permutation(n_mol)
    n_train, n_val = config["n_train"], config["n_val"]
    train_idx = perm[:n_train]
    test_idx_all = perm[n_train + n_val:]

    data = {
        "X_train": x_all[train_idx], "H_train": h_all[train_idx], "M_train": mask_all[train_idx],
        "X_test_all": x_all[test_idx_all], "H_test_all": h_all[test_idx_all], "M_test_all": mask_all[test_idx_all],
    }
    print(f"train={n_train}  test_pool={len(test_idx_all)}", flush=True)
    return data


# ---------------------------------------------------------------------------
# Stability metric (same lookup table as everywhere else in this project) --
# used to filter to stable-only molecules before the memorization comparison.
# ---------------------------------------------------------------------------
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
    return bool(atom_stable.all())


# ---------------------------------------------------------------------------
# Rotation + translation + permutation-invariant geometry comparison: iterative
# Kabsch alignment + Hungarian per-element-type atom assignment (same idea as
# RDKit's GetBestRMS). Only defined between same-formula molecules. A single
# ICP run gets trapped in local optima on molecules with many same-type atoms
# (verified empirically against a synthetic exact-RMSD-0 case), hence the
# random-restart loop, kept cheap via early-stopping once near-zero is found.
# ---------------------------------------------------------------------------
def molecule_formula_key(symbols):
    return tuple(sorted(Counter(symbols).items()))


def _kabsch_rotation(P, Q):
    H = Q.T @ P
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def aligned_rmsd(posA, symA, posB, symB, n_iters=15, n_restarts=100, seed=0, early_stop=1e-6):
    rng = np.random.default_rng(seed)
    idx_by_type_A, idx_by_type_B = defaultdict(list), defaultdict(list)
    for i, s in enumerate(symA):
        idx_by_type_A[s].append(i)
    for i, s in enumerate(symB):
        idx_by_type_B[s].append(i)

    best = np.inf
    for _ in range(n_restarts):
        corr = np.zeros(len(symA), dtype=int)
        for t, a_list in idx_by_type_A.items():
            b_list = idx_by_type_B[t].copy()
            rng.shuffle(b_list)
            for a_i, b_i in zip(a_list, b_list):
                corr[a_i] = b_i

        posB_rot = posB.copy()
        for _ in range(n_iters):
            B_matched = posB_rot[corr]
            R = _kabsch_rotation(posA, B_matched)
            posB_rot = posB_rot @ R.T
            new_corr = corr.copy()
            changed = False
            for t, a_list in idx_by_type_A.items():
                b_list = idx_by_type_B[t]
                A_sub, B_sub = posA[a_list], posB_rot[b_list]
                cost = ((A_sub[:, None, :] - B_sub[None, :, :]) ** 2).sum(-1)
                row_ind, col_ind = linear_sum_assignment(cost)
                for ri, ci in zip(row_ind, col_ind):
                    a_idx, b_idx = a_list[ri], b_list[ci]
                    if new_corr[a_idx] != b_idx:
                        changed = True
                    new_corr[a_idx] = b_idx
            corr = new_corr
            if not changed:
                break

        rmsd = np.sqrt(((posA - posB_rot[corr]) ** 2).sum(1).mean())
        best = min(best, rmsd)
        if best < early_stop:
            break
    return best


def _align_worker(args):
    posA, symA, posB, symB, n_restarts, seed = args
    return aligned_rmsd(posA, symA, posB, symB, n_restarts=n_restarts, seed=seed)


def nearest_neighbor_aligned_rmsd(query_pos, query_syms, candidate_positions, candidate_syms_list,
                                   n_restarts, seed, executor):
    if len(candidate_positions) == 0:
        return np.nan, np.nan
    tasks = [(query_pos, query_syms, cp, cs, n_restarts, seed) for cp, cs in zip(candidate_positions, candidate_syms_list)]
    dists = np.array(list(executor.map(_align_worker, tasks, chunksize=max(1, len(tasks) // (N_WORKERS * 4)))))
    order = np.argsort(dists)
    d1 = float(dists[order[0]])
    d2 = float(dists[order[1]]) if len(dists) > 1 else float("inf")
    return d1, d2


MEMORIZED_RATIO_THRESHOLD = float(os.environ.get("MEMORIZED_RATIO_THRESHOLD", 1.0 / 3.0))


def classify_molecules(n_total, n_stable, d1, d2, ratio_threshold=MEMORIZED_RATIO_THRESHOLD):
    """Partition n_total generated molecules into exactly 3 disjoint, exhaustive buckets:
    broken (failed the stability filter, not in d1/d2 at all), memorized (stable, and
    disproportionately close to one training molecule: d1/d2 < ratio_threshold), good
    (stable, not memorized -- including molecules whose formula has no training match at
    all, since there's nothing to have memorized from)."""
    n_broken = n_total - n_stable
    ratio = d1 / d2   # d1 <= d2 by construction (nearest, second-nearest); ratio in (0, 1]
    no_match = np.isnan(d1)
    is_memorized = (~no_match) & (ratio < ratio_threshold)
    n_memorized = int(is_memorized.sum())
    n_good = n_stable - n_memorized
    return {
        "n_total": n_total, "n_broken": n_broken, "n_stable": n_stable,
        "n_memorized": n_memorized, "n_good": n_good, "n_no_match": int(no_match.sum()),
        "frac_broken": n_broken / n_total, "frac_memorized": n_memorized / n_total, "frac_good": n_good / n_total,
    }


def filter_stable(X, types, mask):
    keep = []
    for b in range(X.shape[0]):
        n = int(mask[b].sum())
        pos = X[b, :n]
        syms = [config["elements"][t] for t in types[b, :n]]
        if molecule_stability(pos, syms):
            keep.append(b)
    return keep


def run_nn_search(query_pos_list, query_syms_list, train_buckets, train_symbols_all, X_train, executor, rng_seed):
    rng = np.random.default_rng(rng_seed)
    d1_list, d2_list, n_cand_list = [], [], []
    for pos, syms in zip(query_pos_list, query_syms_list):
        key = molecule_formula_key(syms)
        candidates = train_buckets.get(key, [])
        if len(candidates) > MAX_CANDIDATES_PER_BUCKET:
            candidates = list(rng.choice(candidates, size=MAX_CANDIDATES_PER_BUCKET, replace=False))
        n_cand_list.append(len(candidates))
        if len(candidates) == 0:
            d1_list.append(np.nan); d2_list.append(np.nan)
            continue
        cand_pos = [X_train[c] for c in candidates]
        cand_syms = [train_symbols_all[c] for c in candidates]
        d1, d2 = nearest_neighbor_aligned_rmsd(pos, syms, cand_pos, cand_syms,
                                                n_restarts=N_RESTARTS, seed=rng_seed, executor=executor)
        d1_list.append(d1); d2_list.append(d2)
    return np.array(d1_list), np.array(d2_list), np.array(n_cand_list)


if __name__ == "__main__":
    data = load_data()
    X_train, H_train, M_train = data["X_train"], data["H_train"], data["M_train"]
    X_test_all, H_test_all, M_test_all = data["X_test_all"], data["H_test_all"], data["M_test_all"]
    N_TRAIN = X_train.shape[0]

    # bucket the training set by formula
    t0 = time.time()
    train_symbols_all = []
    for i in range(N_TRAIN):
        n = int(M_train[i].sum())
        train_symbols_all.append([config["elements"][t] for t in H_train[i, :n].argmax(-1)])
    train_buckets = {}
    for i, syms in enumerate(train_symbols_all):
        train_buckets.setdefault(molecule_formula_key(syms), []).append(i)
    print(f"bucketed {N_TRAIN} training molecules into {len(train_buckets)} formulas in {time.time() - t0:.1f}s"
          f"  (largest bucket: {max(len(v) for v in train_buckets.values())})", flush=True)

    # load the cluster-sampled generated molecules + the exact test_idx used to evaluate them
    samples = torch.load(SAMPLES_PATH, map_location="cpu")
    x_gen, types_gen, mask_gen = samples["x_gen"].numpy(), samples["types_gen"].numpy(), samples["mask_gen"].numpy()
    test_idx = samples["test_idx"].numpy()
    X_test, H_test, M_test = X_test_all[test_idx], H_test_all[test_idx], M_test_all[test_idx]
    test_types = H_test.argmax(-1)
    print(f"loaded {x_gen.shape[0]} generated molecules from {SAMPLES_PATH}", flush=True)

    gen_stable_idx = filter_stable(x_gen, types_gen, mask_gen)
    test_stable_idx = filter_stable(X_test, test_types, M_test)
    print(f"stable generated: {len(gen_stable_idx)}/{x_gen.shape[0]}   "
          f"stable held-out: {len(test_stable_idx)}/{len(test_idx)}", flush=True)

    gen_pos_list = [x_gen[b, :int(mask_gen[b].sum())] for b in gen_stable_idx]
    gen_syms_list = [[config["elements"][t] for t in types_gen[b, :int(mask_gen[b].sum())]] for b in gen_stable_idx]
    test_pos_list = [X_test[b, :int(M_test[b].sum())] for b in test_stable_idx]
    test_syms_list = [[config["elements"][t] for t in test_types[b, :int(M_test[b].sum())]] for b in test_stable_idx]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        gen_d1, gen_d2, gen_ncand = run_nn_search(gen_pos_list, gen_syms_list, train_buckets, train_symbols_all, X_train, ex, rng_seed=0)
        test_d1, test_d2, test_ncand = run_nn_search(test_pos_list, test_syms_list, train_buckets, train_symbols_all, X_train, ex, rng_seed=1)
    print(f"aligned-RMSD search: {len(gen_stable_idx)} generated + {len(test_stable_idx)} held-out queries "
          f"in {time.time() - t0:.1f}s ({N_WORKERS} workers)", flush=True)

    print(f"generated: mean aligned-RMSD to nearest train = {np.nanmean(gen_d1):.4f}  "
          f"(no-match: {np.isnan(gen_d1).sum()}/{len(gen_d1)}, median candidates: {np.median(gen_ncand):.0f})", flush=True)
    print(f"held-out:  mean aligned-RMSD to nearest train = {np.nanmean(test_d1):.4f}  "
          f"(no-match: {np.isnan(test_d1).sum()}/{len(test_d1)}, median candidates: {np.median(test_ncand):.0f})", flush=True)

    # 3-way classification: broken (unstable) / memorized (stable, disproportionately close to
    # one training molecule) / good (stable, not memorized) -- the number to track run over run
    # (e.g. unguided vs. Fisher-Rao-guided sampling).
    gen_class = classify_molecules(x_gen.shape[0], len(gen_stable_idx), gen_d1, gen_d2)
    test_class = classify_molecules(len(test_idx), len(test_stable_idx), test_d1, test_d2)
    print(f"\n[generated]  broken={gen_class['n_broken']} ({gen_class['frac_broken']:.1%})   "
          f"memorized={gen_class['n_memorized']} ({gen_class['frac_memorized']:.1%})   "
          f"good={gen_class['n_good']} ({gen_class['frac_good']:.1%})   "
          f"(ratio threshold={MEMORIZED_RATIO_THRESHOLD:.3f}, no-match={gen_class['n_no_match']})", flush=True)
    print(f"[held-out]   broken={test_class['n_broken']} ({test_class['frac_broken']:.1%})   "
          f"memorized={test_class['n_memorized']} ({test_class['frac_memorized']:.1%})   "
          f"good={test_class['n_good']} ({test_class['frac_good']:.1%})   "
          f"(ratio threshold={MEMORIZED_RATIO_THRESHOLD:.3f}, no-match={test_class['n_no_match']})", flush=True)

    # cast numpy arrays to torch tensors -- newer torch.load defaults to weights_only=True,
    # which rejects raw numpy internals (numpy._core.multiarray._reconstruct) but accepts tensors
    torch.save({
        "gen_d1": torch.from_numpy(gen_d1), "gen_d2": torch.from_numpy(gen_d2),
        "gen_ncand": torch.from_numpy(gen_ncand), "gen_stable_idx": torch.tensor(gen_stable_idx),
        "test_d1": torch.from_numpy(test_d1), "test_d2": torch.from_numpy(test_d2),
        "test_ncand": torch.from_numpy(test_ncand), "test_stable_idx": torch.tensor(test_stable_idx),
        "max_candidates_per_bucket": MAX_CANDIDATES_PER_BUCKET, "n_restarts": N_RESTARTS,
        "memorized_ratio_threshold": MEMORIZED_RATIO_THRESHOLD,
        "gen_classification": gen_class, "test_classification": test_class,
    }, OUT_PATH)
    print(f"saved results to {OUT_PATH}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(gen_d1[~np.isnan(gen_d1)], bins=30, alpha=0.6, density=True, color="steelblue", label="generated")
    axes[0].hist(test_d1[~np.isnan(test_d1)], bins=30, alpha=0.6, density=True, color="darkorange", label="held-out (real)")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="aligned RMSD to nearest same-formula training molecule (A)",
                ylabel="density (log scale)", title="Rigorous (rotation+permutation-invariant) memorization")
    axes[0].legend()

    nn_ratio_gen = gen_d1 / gen_d2     # d(nearest) / d(2nd-nearest); small ratio = disproportionately close to one point
    nn_ratio_test = test_d1 / test_d2
    axes[1].hist(nn_ratio_gen[np.isfinite(nn_ratio_gen)], bins=30, alpha=0.6, density=True, color="steelblue", label="generated")
    axes[1].hist(nn_ratio_test[np.isfinite(nn_ratio_test)], bins=30, alpha=0.6, density=True, color="darkorange", label="held-out (real)")
    axes[1].axvline(MEMORIZED_RATIO_THRESHOLD, color="red", linestyle="--", linewidth=1,
                     label=f"memorized threshold ({MEMORIZED_RATIO_THRESHOLD:.2f})")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="d(1st NN) / d(2nd NN)  (aligned RMSD)", ylabel="density (log scale)", title="Nearest-neighbor ratio")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"saved plot to {PLOT_PATH}", flush=True)
