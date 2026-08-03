# `src/guidance`

Model- and energy-method-agnostic guided-diffusion sampling. Anything that can supply a base reverse step, a
denoiser `y_fn`, and an energy `energy_fn(y) -> I(y)` can run guided sampling through this package. It doesn't
know whether the state is a toy 1D-manifold point, an SO(n) rotation matrix, a QM9 molecule, or a CIFAR-10 image.

## Files

- **`driver.py`** — the shared driver.
  - `guided_reverse_loop` runs the reverse diffusion with a guidance correction spliced in on a schedule.
  - `guidance_grad`/`make_autograd_guidance_fn` — the **unprojected** (ambient) path: exact autograd gradient
    of `I(D_t(z_t))` w.r.t. `z_t`, chaining through the denoiser.
  - `projected_guidance_grad`/`make_projected_guidance_fn` — the **projected** path: differentiates
    `energy_fn(y, P)` directly in y-space (no chain through the denoiser), so both the energy and its
    gradient are tangent-restricted by construction.
  - `eig_tangent_projector`/`make_projector_fn` — dense `[B,D,D]` projector from any per-sample
    Jacobian/metric matrix (eigendecomposition, keep the top-`k` smallest-`|eigenvalue|` directions). Fine at
    small `D` (toy/SO(n)); a dense matrix is intractable at image scale.
  - `make_lowrank_normal_projector_fn` — operator-based projector for a low-rank Jacobian factor `U: [B,D,R]`
    (e.g. image-scale JacNet): projects OUT the top-`r'` normal directions instead of keeping a small-`k`
    tangent subset (the inverse framing — see its docstring for why keeping directions from a low-rank
    matrix's null space would be numerically arbitrary).
  - `apply_projector` — dispatches a projector (dense tensor or callable) uniformly; every energy module
    below goes through this rather than applying `P` directly.

- **`basic_fr.py`** — Basic algorithm variant. k-NN Fisher-Rao energy: a softmax posterior over the k nearest
  training atoms, temperature pinned to `sigma_t^2` (no free hyperparameter to tune).

- **`score_divergence.py`** — Empirical score-divergence variant. `I(y) = trace(grad_y s_theta(y,t)) +
  d/sigma_t^2`, estimated with Hutchinson probes; needs no training-atom index. Has both an exact
  (double-backward) and finite-difference (SPSA) gradient path — see its docstring for the memory tradeoff  between them.

- **`bayesian_fr.py`** — Bayesian variant, **stub only**. Meant to be a Dirichlet-bootstrap posterior over
  `basic_fr`'s k-NN energy (replace the fixed uniform prior over neighbors with `B` Dirichlet(α) draws,
  averaged). Reference implementation to port from:
  `preliminary/fisher_rao_guidance_so(n).ipynb`'s `soft_posterior_bb`/`fisher_rao_energy_bb` cell.

## Unprojected vs. projected

`make_autograd_guidance_fn(y_fn, energy_fn, ...)` (unprojected) vs. `make_projected_guidance_fn(y_fn, energy_fn,
projector_fn, ...)` (projected) — not one function with a `None`-able projector, because the two compute
genuinely different gradients (see `projected_guidance_grad`'s docstring): chaining the unprojected gradient
through the denoiser and then reprojecting would conflate the denoiser's own Jacobian with the data manifold's
tangent geometry.

Three ways to build a `projector_fn(y) -> P`, all used somewhere in this repo:

- **Known analytic tangent** (`experiments/toy_manifold/02_fr_guidance.ipynb`): write it directly, e.g.
  `lambda y: torch.einsum("bi,bj->bij", t(y), t(y))`.
- **Learned dense Jacobian** (`experiments/so_n/03_fr_guidance.ipynb`, via a trained JacNet):
  `driver.make_projector_fn(jacobian_matrix_fn, k=...)`, `jacobian_matrix_fn(y) -> M [B,D,D]`. `k` is an
  explicit manifold-dimension ablation parameter.
- **Learned low-rank Jacobian** (`cluster-codes/cifar10/sample_cifar10_fr_guidance.py`, image scale):
  `driver.make_lowrank_normal_projector_fn(u_fn, r_prime)`, `u_fn(y) -> U [B,D,R]`.

