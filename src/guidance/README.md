# `src/guidance`

Model- and energy-method-agnostic guided-diffusion sampling. Anything that can supply a base reverse step, a
denoiser `y_fn`, and an energy `energy_fn(y) -> I(y)` can run guided sampling through this package. It doesn't
know whether the state is a toy 1D-manifold point, a QM9 molecule, or a CIFAR-10 image.

## Files

- **`driver.py`** — the shared driver.
  - `guided_reverse_loop` runs the reverse diffusion with a guidance correction spliced in on a schedule,
    strength set by `trust_region` (`eta`, a float or callable `eta_fn(step_idx)`) — see
    `guidance_correction` for the relative parameterization, `correction = eta * ||base step|| *
    grad/||grad||`. This is the only guidance-strength mechanism now; the older absolute `lam *
    step_scale_fn(t)` path and per-gradient `grad_clip` were removed (some `cluster-codes/**` scripts
    still target that old API — to be fixed).
  - `guidance_grad`/`make_autograd_guidance_fn` — the **unprojected** (ambient) path: exact autograd gradient
    of `I(D_t(z_t))` w.r.t. `z_t`, chaining through the denoiser.
  - `projected_guidance_grad`/`make_projected_guidance_fn` — the **projected** path: differentiates
    `energy_fn(y, P)` directly in y-space (no chain through the denoiser), so both the energy and its
    gradient are tangent-restricted by construction.
  - `eig_tangent_projector`/`make_projector_fn` — dense `[B,D,D]` projector from any per-sample
    Jacobian/metric matrix (eigendecomposition, keep the top-`k` smallest-`|eigenvalue|` directions). Fine at
    small `D` (e.g. the toy manifold); a dense matrix is intractable at image scale. The image-scale,
    low-rank counterpart (`make_lowrank_normal_projector_fn`, operating on a JacNet factor `U: [B,D,R]`)
    was retired along with the JacNet training scripts it depended on.
  - `apply_projector` — dispatches a projector (dense tensor or callable) uniformly; every energy module
    below goes through this rather than applying `P` directly.

- **`basic_fr.py`** — the settled recipe. k-NN Fisher-Rao energy: a softmax posterior over the k nearest
  training atoms, outer scale pinned to `sigma_t^2` (no free hyperparameter). `q_target` (via
  `beta_soft_for`/`beta_for_target_qmax`) adaptively tempers the softmax so its gradient stays informative
  even where the untempered posterior would saturate to one-hot (measured to matter at CIFAR-10 scale,
  `D=3072`; a no-op by construction on low-dimensional toy manifolds).

- **`bayesian_fr.py`** / **`bayesian_fr_v2.py`** — active research, not the settled recipe.
  `bayesian_fr.py` replaces `basic_fr`'s fixed uniform prior over the k retrieved atoms with a
  Dirichlet-bootstrap posterior (`B` Dirichlet(α) draws, averaged). `bayesian_fr_v2.py` is the
  closed-form counterpart (a Dirichlet posterior on the assignment itself, no sampling). Exercised by
  `experiments/toy_manifold/04_bayesian_fr_1d_slice.ipynb` and `05_bayesian_regime_comparison.ipynb`.

`score_divergence.py` (the empirical score-divergence energy, `I(y) = trace(grad_y s_theta(y,t)) +
d/sigma_t^2`) was retired — degrades performance when the tangent space is unknown.

## Unprojected vs. projected

`make_autograd_guidance_fn(y_fn, energy_fn, ...)` (unprojected) vs. `make_projected_guidance_fn(y_fn, energy_fn,
projector_fn, ...)` (projected) — not one function with a `None`-able projector, because the two compute
genuinely different gradients (see `projected_guidance_grad`'s docstring): chaining the unprojected gradient
through the denoiser and then reprojecting would conflate the denoiser's own Jacobian with the data manifold's
tangent geometry.

Two ways to build a `projector_fn(y) -> P`, both used in this repo:

- **Known analytic tangent** (`experiments/toy_manifold/02_fr_guidance.ipynb`, `JACOBIAN_TYPE="closed-form"`):
  write it directly, e.g. `lambda y: torch.einsum("bi,bj->bij", t(y), t(y))`.
- **Learned/exact dense Jacobian**: `driver.make_projector_fn(jacobian_matrix_fn, k=...)`,
  `jacobian_matrix_fn(y) -> M [B,D,D]`. `k` is an explicit manifold-dimension ablation parameter. Used with
  the real per-sample autodiff Jacobian in `experiments/toy_manifold/02_fr_guidance.ipynb`
  (`JACOBIAN_TYPE="auto-diff"`, cheap at `d=2`).

The image-scale, low-rank projector path (`driver.make_lowrank_normal_projector_fn`, consumed by
`cluster-codes/cifar10/sample_cifar10_fr_guidance.py`) is retired — that script needs migration before
it will run.

