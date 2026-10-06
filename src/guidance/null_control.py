"""Random perturbations, the null control for FR: a random unit direction with FR's step size and window.

Returns guidance_fn(z_t) -> (grad [B, D], I [B], in_range [B]) for driver.guided_reverse_loop.
"""
import torch


def _random_unit(z_t, generator=None):
    """Random unit rows shaped like z_t, drawn on the generator's device; a list gives one generator per row."""
    if isinstance(generator, (list, tuple)):
        r = torch.cat([torch.randn((1, *z_t.shape[1:]), dtype=z_t.dtype, device=g.device, generator=g)
                       for g in generator]).to(z_t.device)
        return r / r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    device = generator.device if generator is not None else z_t.device
    r = torch.randn(z_t.shape, dtype=z_t.dtype, device=device, generator=generator).to(z_t.device)
    r = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return r


def make_null_control_guidance_fn(generator=None):
    """Random unit direction, I = 0, all in range; with eta-mode the correction magnitude matches FR guidance."""
    def guidance_fn(z_t):
        B = z_t.shape[0]
        r = _random_unit(z_t, generator=generator)
        I = torch.zeros(B, dtype=z_t.dtype, device=z_t.device)
        in_range = torch.ones(B, dtype=torch.bool, device=z_t.device)
        return r, I, in_range
    return guidance_fn

