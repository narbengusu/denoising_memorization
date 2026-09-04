# `src/guidance`

Model- and energy-agnostic guided-diffusion sampling. Anything that supplies a base reverse
step, a denoiser `y_fn`, and an energy `energy_fn(y) -> I(y)` can run guided sampling through
this package.

## Files

- **`driver.py`**, the shared driver.
  - `guided_reverse_loop` splices a guidance correction into the reverse diffusion on a
    schedule, strength set by `trust_region` (`eta`, a float or `eta_fn(step_idx)`). See
    `guidance_correction`: `correction = eta * ||base step|| * grad/||grad||`.
  - `guidance_grad` / `make_autograd_guidance_fn`: unprojected (ambient) path, exact autograd
    gradient of `I(D_t(z_t))` through the denoiser.
  - `projected_guidance_grad` / `make_projected_guidance_fn`: projected path, differentiates
    `energy_fn(y, P)` directly in y-space so the gradient is tangent-restricted by
    construction.
  - `eig_tangent_projector` / `make_projector_fn`: dense `[B,D,D]` projector from a per-sample
    Jacobian/metric matrix. Fine at small `D` (toy manifold); intractable at image scale.
  - `apply_projector`: dispatches a projector (dense tensor or callable) uniformly.

- **`basic_fr.py`**, the settled recipe. k-NN Fisher-Rao energy: softmax posterior over the k
  nearest training atoms, outer scale pinned to `sigma_t^2`. `q_target` (via `beta_soft_for` /
  `beta_for_target_qmax`) adaptively tempers the softmax so its gradient stays informative
  where the untempered posterior would saturate (matters at CIFAR-10 scale, `D=3072`; a no-op
  on low-dimensional manifolds).

- **`bayesian_fr.py`**, the Bayesian FR variant reported alongside `basic_fr` in the paper
  figures. Replaces `basic_fr`'s fixed uniform prior with a Dirichlet-bootstrap posterior (`B`
  Dirichlet draws, averaged). Used directly by `workshop_figures/generate/*`.

## Unprojected vs. projected

`make_autograd_guidance_fn` (unprojected) and `make_projected_guidance_fn` (projected) compute
genuinely different gradients, so they are two functions rather than one with an optional
projector. Chaining the unprojected gradient through the denoiser and then reprojecting would
conflate the denoiser's own Jacobian with the data manifold's tangent geometry.

Two ways to build a `projector_fn(y) -> P`:

- Known analytic tangent (`experiments/sinusoid/02_fr_guidance.ipynb`,
  `JACOBIAN_TYPE="closed-form"`): write it directly.
- Learned/exact dense Jacobian: `driver.make_projector_fn(jacobian_matrix_fn, k=...)`. Used
  with the real per-sample autodiff Jacobian in the same notebook
  (`JACOBIAN_TYPE="auto-diff"`, cheap at `d=2`).

## Retired

Not part of the settled recipe, moved to `preliminary/`: `score_divergence.py`,
`train_cifar10_jacnet.py`, `train_qm9_jacnet.py`, `driver.make_lowrank_normal_projector_fn`
(the image-scale low-rank tangent projector), `basic_fr.make_knn_bisector_scale_fn`, and
`bayesian_fr_v2.py` (the closed-form Bayesian FR variant, not reported; `bayesian_fr.py`'s
bootstrap variant is the one used in the paper).
