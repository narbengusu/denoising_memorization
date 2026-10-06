# CelebA-HQ, latent EDM

EDM in the Stable Diffusion VAE latent space (4x32x32 latents of 256x256 faces), trained on all of CelebA-HQ and fine-tuned on 200 images until it memorizes. 32-step Heun sampler. Code: `src/models/celeba_hq_edm.py`.

**Setup:** the 30k CelebA-HQ images in `data/celeba_hq/images/` (manual download, see `src/data/celeba_hq.py`). Run scripts from this folder.

| Step | Does | Writes | Time |
|---|---|---|---|
| `01_download_vae.py` | Downloads `stabilityai/sd-vae-ft-mse` | `pretrained/` | minutes |
| `02_precompute_latents.py` | Encodes every image and its mirror | `checkpoints/celeba_hq_latents.pt` | GPU |
| `03_train.py` | Trains the EDM on all latents | `checkpoints/celeba_hq_edm.pt` | days (GPU) |
| `04_finetune.py` | Fine-tunes on 200 latents, 50k iterations | `checkpoints/celeba_hq_edm_finetuned.pt` | GPU |
| `05_guidance_window.py` | FR diagnostics along unguided trajectories (justifies the window) | `all_figures/supp_guidance_window.pdf` | ~1 min |
| `06_benchmark.ipynb` | Every method on 5 memorized and 5 non-memorized seeds | `all_figures/celeba_hq_{memorized,non_memorized}.pdf`, `results/per_seed.csv` | ~40 min (MPS) |

**Settings**
- Memorized: $d_1/d_2 < 1/3$ against the 200 training latents.
- Seeds: memorized 22, 20, 21, 11, 15; non-memorized 125, 75, 87, 85, 88.
- FR: k = 15, window 0.35-1, $\eta = 0.2$, $q^* = 0.8$. Bayesian FR: $\alpha_D = 0.2$, B = 256.
- AMG: EDM form, nearest neighbour searched in pixel space; $c_3$ = the smallest value in {0.1, 0.3, 1, 3, 10} that moves the most memorized seeds out of memorization.
- Broken Memories: best setting of a one-parameter-at-a-time sweep; the "paper" and $\tau$-scale rows keep the published k = 10 steps (of 32 here).
