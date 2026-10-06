# CIFAR-10, OpenAI's pretrained iDDPM

OpenAI's unconditional CIFAR-10 iDDPM (the model AMG uses), 250 respaced DDPM steps. Code: `src/models/iddpm.py`, which clones `improved-diffusion` into `external/` on first use.

**Setup**
- CIFAR-10 in `data/cifar-10-batches-py/`.
- `cifar10_uncond_50M_500K.pt` from [improved-diffusion](https://github.com/openai/improved-diffusion) in `pretrained/`.

| Step | Does | Writes | Time (MPS) |
|---|---|---|---|
| `00_parameter_tuning.ipynb` | Picks FR's window, then a grid over $q^*$, $\eta$ on held-out seeds | `cache/fr_settings.json` | ~10 min |
| `01_benchmark.ipynb` | Screens 2000 seeds, runs every method on 5 memorized and 5 non-memorized seeds | `all_figures/cifar10_openai_{memorized,non_memorized}.pdf`, `results/` | ~45 min (after the first screen) |

**Settings**
- Memorized: nL2 < 1.4 against the 50k training images (AMG's criterion). This model makes near-copies, not exact ones.
- Seeds: memorized 1755, 1527, 209, 1635, 542; non-memorized 34, 40, 86, 89, 92.
- FR: window 0.1-1, $q^* = 0.9$, $\eta = 0.1$.
- Broken Memories: best setting of a one-parameter-at-a-time sweep; the "paper" and $\tau$-scale rows keep the published k = 10 steps (of 250 here).
- AMG: DDPM form, $c$ = the smallest value in {0.3, 1, 3, 10} that moves the most memorized seeds out of memorization.
