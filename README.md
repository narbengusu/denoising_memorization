# Fisher-Rao guidance against memorization in diffusion models

```
src/
  guidance/               FR, Bayesian FR, Random perturbations, guided reverse loop
  baselines/              AMG, Broken Memories
  models/                 per-model loading, samplers and metrics
  data/                   CIFAR-10 and CelebA-HQ loaders
  plotting/wsstyle.py     paper style: fonts, sizes, method names and colours
experiments/              one folder per experiment, numbered steps, each with a README
supplementary_figures/    model-free figures
all_figures/              every paper figure (PDF)
```

## Reproducing

```
pip install -r requirements.txt
```

Run each experiment's steps in order. Experiments are independent of each other.

- [`sinusoid`](experiments/sinusoid/README.md)
- [`cifar10_openai`](experiments/cifar10_openai/README.md)
- [`celeba_hq`](experiments/celeba_hq/README.md)

The supplementary figures are produced by `python supplementary_figures/finite_sample_resolution.py` and `python supplementary_figures/fisher_rao_field.py`.

`DEVICE` selects `cpu`, `mps` or `cuda` (default: whichever GPU is available). Each seed has its own random streams, so results do not depend on device or batch size, up to floating-point rounding.

Weights (`checkpoints/`, `pretrained/`), run caches (`cache/`) and CSVs (`results/`) are written inside each experiment folder and are gitignored. Steps resume from `cache/`.

## Figures

| `all_figures/` | Made by |
|---|---|
| `sinusoid_pareto_N20.pdf`, `sinusoid_final_positions_N20.pdf` | `experiments/sinusoid/02_benchmark.ipynb` |
| `supp_bayesian_convergence.pdf` | `experiments/sinusoid/03_bayesian_convergence.ipynb` |
| `cifar10_openai_{memorized,non_memorized}.pdf` | `experiments/cifar10_openai/01_benchmark.ipynb` |
| `celeba_hq_{memorized,non_memorized}.pdf` | `experiments/celeba_hq/06_benchmark.ipynb` |
| `supp_guidance_window.pdf` | `experiments/celeba_hq/05_guidance_window.py` |
| `supp_finite_sample_resolution.pdf` | `supplementary_figures/finite_sample_resolution.py` |
| `supp_fisher_rao_field.pdf` | `supplementary_figures/fisher_rao_field.py` |
