# SO(n) Diffusion — Agent Instructions

## Theme
Train a VP-SDE score model on SO(n) rotation matrices treated as flat vectors in R^{n²}. No manifold structure is given to the model. The research question: does the model learn SO(n) implicitly?

## Goal
Implement a single Jupyter notebook (`so_n_diffusion.ipynb`) that:
1. Generates a frozen target set of SO(n) matrices (Haar-sampled, optionally perturbed around K modes)
2. Trains a noise-prediction MLP via VP-SDE denoising score matching
3. Samples with reverse-time Euler-Maruyama (no projection step)
4. Evaluates via orthogonality residual, determinant histogram, memorization test, and (n=3) rotation angle histogram

## Code style rules
- **No classes** except a single `nn.Module` for the score network
- Short, readable, flat code — one notebook top to bottom
- Freely reshape between `[B, n, n]` (matrix view) and `[B, n²]` (model view)
- No comments unless the WHY is non-obvious

## Key implementation notes
- `sample_haar(n, B)`: QR of Gaussian, fix column signs, flip last column so det=+1
- Target set: `sample_haar(n, N_data).reshape(N_data, d)` — Haar samples are exact SO(n) members, no further processing needed
- VP schedule: `beta(t) = beta_min + t*(beta_max-beta_min)`, `alpha_bar(t) = exp(-beta_min*t - 0.5*(beta_max-beta_min)*t²)`
- Clamp `t >= 1e-3` in loss to avoid `1/(1-alpha_bar)` instability
- Sampler output with untrained net should be small/random — if it diverges, the drift sign is wrong
- Eval projection (SVD-based) is for visualization only, never used in the sampler

## Config
All hyperparameters live in a single `config` dict. No `sigma_data` or `K_modes` — target is plain Haar. Device comes from `.env` via `os.environ.get("DEVICE", "cpu")`.

## Build order
Follow `plan.md` top to bottom. Verify each phase before moving on. Two run modes: memorization (N_data=10) and distributional (N_data=5000).

## Ask before
- Changing network architecture beyond what manual.md specifies
- Adding projection or manifold-aware components
- Splitting into multiple files
- Running something in the background. Do not run neural-network train files in the background. They are computationally expensive, ideally we would use the cluster. 
