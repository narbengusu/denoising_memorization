
## Structure

```
src/guidance/         shared guided-sampling library, used by every experiment
experiments/           one folder per dataset: download, finetune, sample, guide
  sinusoid/               1-D curve in R^2, no external data
  cifar10/                 finetuned pixel-space DDPM
  celeba_hq/                finetuned latent-space EDM
workshop_figures/      one generator per paper figure, rendered output in out/
data/                  downloaded datasets (gitignored)
checkpoints/, pretrained/, trained_data/
                        model weights and run artifacts (gitignored, regenerable)
preliminary/           exploratory work not reported in the paper, not maintained
```

## Replication guide

```
pip install -r requirements.txt
```

**Sinusoid**: no download needed. Run `experiments/sinusoid/01_train_score.ipynb`
through `03_fisher_rao_geometry.ipynb` in order.

**CIFAR-10**:
1. `python experiments/cifar10/download_model.py` (fetches `google/ddpm-cifar10-32`)
2. `python experiments/cifar10/finetune_cifar10_ddpm.py` (CIFAR-10 itself downloads
   automatically on first import; finetunes on a 1000-image subset until it memorizes)
3. `01_generate_memorized_sample.ipynb` then `02_fr_guidance.ipynb`

**CelebA-HQ**:
1. `python experiments/celeba_hq/download_vae.py` (fetches `stabilityai/sd-vae-ft-mse`)
2. Place the 30k CelebA-HQ `.jpg` files at `data/celeba_hq/images/` (no scriptable download,
   see `celeba_hq_data.py`)
3. `python experiments/celeba_hq/precompute_latents.py`
4. `python experiments/celeba_hq/finetune_celeba_hq_edm.py` (finetunes on a 200-latent subset)
5. `01_sample_celeba_hq.ipynb` then `02_fr_guidance.ipynb`

**Figures**: once an experiment's checkpoints exist, run the numbered generators in
`workshop_figures/generate/`. See `workshop_figures/README.md`.

`DEVICE` (env var, default `cpu`) selects `cpu`/`mps`/`cuda` throughout.
