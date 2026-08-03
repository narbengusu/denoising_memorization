import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from assess_memorization_qm9 import (
    config, load_data, filter_stable, molecule_formula_key, run_nn_search, classify_molecules,
    MEMORIZED_RATIO_THRESHOLD, N_RESTARTS, N_WORKERS,
)

# Compares the unguided baseline against BOTH guidance methods across LAMBDA_LIST:
# samples_fr_lambda{L}.pt (Algorithm 1, k-NN Fisher-Rao energy, from sample_qm9_edm_fr.py)
# and samples_scorevar_lambda{L}.pt (Algorithm 2, empirical score-divergence variant, from
# sample_qm9_edm_scorevar.py). Runs missing on disk are skipped (e.g. if you only ran one
# method's sampling job) rather than erroring.
LAMBDA_LIST = [float(x) for x in os.environ.get("LAMBDA_LIST", "1,5,10,50,100").split(",")]
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", "checkpoints")
BASELINE_SAMPLES_PATH = os.environ.get("BASELINE_SAMPLES_PATH", "checkpoints/samples.pt")
OUT_PATH = os.environ.get("OUT_PATH", "checkpoints/memorization_fr_sweep.pt")
PLOT_PATH = os.environ.get("PLOT_PATH", "checkpoints/memorization_fr_sweep.png")

# method label -> filename prefix (checkpoints/{prefix}_lambda{L}.pt)
METHODS = {"fr": "samples_fr", "scorevar": "samples_scorevar"}


def classify_samples(samples_path, train_buckets, train_symbols_all, X_train, executor):
    samples = torch.load(samples_path, map_location="cpu")
    x_gen, types_gen, mask_gen = samples["x_gen"].numpy(), samples["types_gen"].numpy(), samples["mask_gen"].numpy()
    stable_idx = filter_stable(x_gen, types_gen, mask_gen)
    pos_list = [x_gen[b, :int(mask_gen[b].sum())] for b in stable_idx]
    syms_list = [[config["elements"][t] for t in types_gen[b, :int(mask_gen[b].sum())]] for b in stable_idx]
    d1, d2, _ = run_nn_search(pos_list, syms_list, train_buckets, train_symbols_all, X_train, executor, rng_seed=0)
    return classify_molecules(x_gen.shape[0], len(stable_idx), d1, d2)


if __name__ == "__main__":
    data = load_data()
    X_train, H_train, M_train = data["X_train"], data["H_train"], data["M_train"]
    N_TRAIN = X_train.shape[0]

    t0 = time.time()
    train_symbols_all = []
    for i in range(N_TRAIN):
        n = int(M_train[i].sum())
        train_symbols_all.append([config["elements"][t] for t in H_train[i, :n].argmax(-1)])
    train_buckets = {}
    for i, syms in enumerate(train_symbols_all):
        train_buckets.setdefault(molecule_formula_key(syms), []).append(i)
    print(f"bucketed {N_TRAIN} training molecules into {len(train_buckets)} formulas in {time.time() - t0:.1f}s",
          flush=True)

    runs = {}
    if os.path.exists(BASELINE_SAMPLES_PATH):
        runs["unguided"] = BASELINE_SAMPLES_PATH
    for method, prefix in METHODS.items():
        for lam in LAMBDA_LIST:
            path = os.path.join(SAMPLES_DIR, f"{prefix}_lambda{lam:g}.pt")
            if os.path.exists(path):
                runs[f"{method}:lambda={lam:g}"] = path
            else:
                print(f"missing {path}, skipping", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for name, path in runs.items():
            t0 = time.time()
            cls = classify_samples(path, train_buckets, train_symbols_all, X_train, ex)
            print(f"[{name}]  broken={cls['n_broken']} ({cls['frac_broken']:.1%})   "
                  f"memorized={cls['n_memorized']} ({cls['frac_memorized']:.1%})   "
                  f"good={cls['n_good']} ({cls['frac_good']:.1%})   "
                  f"(ratio threshold={MEMORIZED_RATIO_THRESHOLD:.3f}, no-match={cls['n_no_match']}, "
                  f"{time.time() - t0:.1f}s)", flush=True)
            results[name] = cls

    torch.save({"results": results, "memorized_ratio_threshold": MEMORIZED_RATIO_THRESHOLD,
                "n_restarts": N_RESTARTS}, OUT_PATH)
    print(f"saved results to {OUT_PATH}", flush=True)

    names = list(results.keys())
    frac_broken = [results[n]["frac_broken"] for n in names]
    frac_memorized = [results[n]["frac_memorized"] for n in names]
    frac_good = [results[n]["frac_good"] for n in names]

    fig, ax = plt.subplots(figsize=(1.5 * len(names) + 3, 4))
    x = np.arange(len(names))
    ax.bar(x, frac_broken, label="broken", color="firebrick")
    ax.bar(x, frac_memorized, bottom=frac_broken, label="memorized", color="darkorange")
    bottom2 = [b + m for b, m in zip(frac_broken, frac_memorized)]
    ax.bar(x, frac_good, bottom=bottom2, label="good", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("fraction of generated samples")
    ax.set_title("Guidance method comparison: broken / memorized / good")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"saved plot to {PLOT_PATH}", flush=True)
