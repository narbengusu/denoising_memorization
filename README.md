## Goal

Can a guidance signal derived from the score's own local geometry (Fisher-Rao energy over the k
nearest training atoms) push samples away from memorized training points without distorting the
rest of the distribution on a learned diffusion model? The settled recipe is **ambient k-NN
Fisher-Rao energy + adaptive softmax temperature (`q_target`) + relative trust region (`eta`)** —
see `src/guidance/README.md` for the full API and its migration status.

## Folder structure

- **`src/guidance/`** — the shared, model-agnostic guided-sampling library every experiment below
  imports. See `src/guidance/README.md` for its API.
- **`experiments/`** — small-scale notebooks, run locally.
  - `toy_manifold/`: a 1D curve in R^2. `01_train_score.ipynb` trains the score net;
    `02_fr_guidance.ipynb` runs the settled recipe's projected-vs-unprojected jacobian comparison
    (`basic_fr` only, relative trust region `eta`, no Bayesian variants); `04_bayesian_fr_1d_slice.ipynb`
    and `05_bayesian_regime_comparison.ipynb` are active research notebooks for the Bayesian
    posterior variants (`bayesian_fr.py`/`bayesian_fr_v2.py`) that `02` no longer covers.
  - `cifar10/`: FR-guidance on the finetuned CIFAR-10 DDPM, mirroring `toy_manifold/02` at image
    scale — `01_generate_memorized_sample.ipynb` picks a memorized seed, `02_fr_guidance.ipynb` runs
    the guided sweep (`q_target` × `eta` × guidance window).
- **`cluster-codes/`** — the same pipeline at real-model scale, as Slurm-submitted scripts rather
  than notebooks (`submit_*.sh`, one per `*.py`).
  - `qm9/`: molecule generation (EGNN).
  - `cifar10/`: image generation (`google/ddpm-cifar10-32`, finetuned to memorize a small subset).
    `sample_cifar10_fr.py` is the basic ambient FR sweep; `sample_cifar10_fr_guidance.py` adds the
    JacNet-projected comparison. Several scripts here still target the pre-cleanup `driver.py` API
    (`lam`/`step_scale_fn`/`grad_clip`, plus JacNet training) - to be fixed. 
- **`checkpoints/`, `data/`, `figures/`, `cluster-codes/*/pretrained/`** — generated artifacts
  (model weights, datasets, plots). Gitignored; regenerate by running the corresponding
  notebook/script.