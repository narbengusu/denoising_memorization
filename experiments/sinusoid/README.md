# Sinusoid

2-D toy: $N = 20$ training points on $y = 0.6 \sin(2x)$, a small VP-SDE score net, 200-step Euler-Maruyama sampler. Code: `src/models/sinusoid.py`. CPU only.

| Step | Does | Writes | Time |
|---|---|---|---|
| `01_train_score.ipynb` | Trains the score net (`N_DATA`, default 20) | `checkpoints/score_normal_N20.pt` | minutes |
| `02_benchmark.ipynb` | Sweeps every method's strength, picks operating points | `all_figures/sinusoid_{pareto,final_positions}_N20.pdf`, `results/summary.csv`, `cache/operating_points.json` | ~20 min |
| `03_bayesian_convergence.ipynb` | 1-D slice of the FR energy, Bayesian FR over $\alpha_D$ | `all_figures/supp_bayesian_convergence.pdf` | seconds |

**Settings**
- Memorized: $d_1/d_2 < 1/3$; training points closer than 0.05 count as one.
- Fidelity: mean squared distance to the curve.
- FR: k = 5, $q^* = 0.9$, window = last 20% of sampling. Bayesian FR: $\alpha_D = 1$ (benchmark sweeps 0.5, 1, 2), B = 256. Random perturbations: FR's $\eta$ and window.
- Broken Memories: the "paper" setting keeps the published k = 10 steps (of 200 here).
- Operating point: the setting whose memorized fraction is closest to 0.2, ties broken by fidelity.
