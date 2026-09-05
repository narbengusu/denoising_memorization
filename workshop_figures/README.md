# Workshop paper figures

Everything that produces a figure for the paper, plus the rendered assets. One generator per
manuscript figure: `figN.ipynb`/`figN.py` writes `out/figN.{pdf,png}`.

```
workshop_figures/
  wsstyle.py     shared house style: rcParams, palette, layout helpers
  generate/       one generator per manuscript figure (figN.ipynb / figN.py)
  out/            rendered figures only (.pdf for LaTeX, .png for viewing)
  cache/          per-run sample caches (*.pt), regenerable, gitignored
```

## Generators

| Generator | Produces | Manuscript figure | Notes |
|---|---|---|---|
| `fig1.ipynb` | `fig1` | Fig. 1 | CelebA-HQ inference-time mitigation, 5 main seeds only. 32-step EDM sampler, a few seconds/run. |
| `fig2.py` | `fig2` | Fig. 2 | Finite-sample resolution: zoom chain on a curved surface. Standalone, no repo deps beyond `wsstyle`. ~15s. |
| `fig3.py` | `fig3` | Fig. 3 | Fisher-Rao fields across bandwidths, same surface as `fig2.py` (shared setup in `_surface_common.py`). ~15s. |
| `fig4.ipynb` | `fig4` | Fig. 4 | Convergence of `I` to FR as `alpha_D` increases. 1-D energy slice over the sinusoid's N=10 atoms; no model, no sampling. |
| `fig5.py` | `fig5` | Fig. 5 | The guidance window: diagnostics behind the CelebA-HQ window/temperature choice. |
| `fig6.ipynb` | `fig6` | Fig. 6 | Sinusoid manifold: Fisher-Rao guidance vs. atomic collapse, with the energy field and cell boundaries. Regenerates from scratch in ~20s. |
| `fig7.ipynb` | `fig7` | Fig. 7 | CIFAR-10 inference-time mitigation, 5 seeds x 3 arms on the finetuned DDPM. ~150s/run, per-run cache. |
| `fig7_alpha_sweep.py` | `fig7_alpha*` | (appendix) | Dirichlet-alpha sweep for `fig7`'s Bayesian arm; reuses `fig7`'s cache. |

## Caches

`cache/*_runs.pt` hold sampled results so re-styling a figure never re-samples. Keyed per run,
so retuning one arm's hyperparameter only re-runs that arm. Delete a cache to force a full
re-run.

## Gitignore

`.gitignore` negates the blanket `*.png`/`checkpoints/` rules for `workshop_figures/out/`, so
the rendered figures stay tracked. `workshop_figures/cache/` stays ignored.
