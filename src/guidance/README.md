# `src/guidance`

Model-agnostic guided sampling. Supply a base reverse step, a denoiser `y_fn`, and an energy `energy_fn(y) -> I(y)`.

| File | Contents |
|---|---|
| `driver.py` | `guided_reverse_loop`: adds a guidance correction to each reverse step inside a window. Correction is `eta * ‖base step‖ * grad/‖grad‖` (`trust_region`), or `lam * grad` if `lam` is given. Gradient builders: `make_autograd_guidance_fn` (ambient), `make_projected_guidance_fn` (tangent-projected, small `D` only). |
| `basic_fr.py` | Fisher-Rao energy over the k nearest training points; `q_target` tempers the softmax so it does not saturate. |
| `bayesian_fr.py` | Bayesian FR: the uniform prior replaced by a Dirichlet bootstrap (`B` draws). |
| `null_control.py` | Random perturbations: FR's step size and window, random direction. |

The paper uses `trust_region` only. The `lam`, `grad_clip` and tangent-projection paths are not used in any reported result.

Pass one `torch.Generator` per sample (a list) to `bayesian_fr` and `null_control` so each sample's randomness is independent of its batch.
