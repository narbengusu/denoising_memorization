## Goal

Can a guidance signal derived from the score's own local geometry (Fisher-Rao energy /
score-divergence, optionally restricted to an estimated tangent subspace) push samples away from
memorized training points without distorting the rest of the distribution on a learned diffusion model?

## Folder structure

- **`src/guidance/`** — the shared, model-agnostic guided-sampling library every experiment below
  imports. See `src/guidance/README.md` for its API.
- **`experiments/`** — small-scale notebooks, run locally.
  - `toy_manifold/`: a 1D curve in R^2. `01_train_score.ipynb` trains the score net;
    `02_fr_guidance.ipynb` runs the full FR-guidance sweep (diagnostic, method × jacobian × window
    × grad_clip × lam, per-parameter marginals).
  - `so_n/`: SO(n) rotation matrices. `01_train_score.ipynb` trains the score net,
    `02_train_jacnet.ipynb` trains a JacNet tangent estimator, `03_fr_guidance.ipynb` runs the same
    sweep shape as `toy_manifold`, projected via the trained JacNet instead of an analytic tangent.
- **`cluster-codes/`** — the same pipeline at real-model scale, as Slurm-submitted scripts rather
  than notebooks (`submit_*.sh`, one per `*.py`).
  - `qm9/`: molecule generation (EGNN).
  - `cifar10/`: image generation (`google/ddpm-cifar10-32`, finetuned to memorize a small subset).
    Order: `submit_cifar10_finetune.sh` → `submit_cifar10_train_jacnet.sh` (low-rank JacNet, see its
    module docstring for why low-rank at image scale) → `submit_cifar10_fr_guidance.sh` (unified
    sweep) → `assess_memorization_cifar10.py` for real FID on a chosen result.
- **`preliminary/`** — earlier, superseded exploratory notebooks. Kept for reference (e.g. the
  Dirichlet-bootstrap derivation `src/guidance/bayesian_fr.py` still needs porting from), not
  maintained currently.
- **`checkpoints/`, `data/`, `figures/`, `cluster-codes/*/pretrained/`** — generated artifacts
  (model weights, datasets, plots). Gitignored; regenerate by running the corresponding
  notebook/script.

## Getting started

1. Train a score net: `experiments/so_n/01_train_score.ipynb` or
   `experiments/toy_manifold/01_train_score.ipynb` (small, local) — or `cluster-codes/*/submit_*train*.sh`
   (cluster-scale).
2. If you want tangent-projected guidance, train JacNet next (`experiments/so_n/02_train_jacnet.ipynb`
   or `cluster-codes/*/submit_*_train_jacnet.sh`).
3. Run the FR-guidance sweep (`*_fr_guidance.ipynb` / `submit_*_fr_guidance.sh`).
