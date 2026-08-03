"""Energy function for the Bayesian variant (Dirichlet-bootstrap posterior
over basic_fr's k-NN Fisher-Rao energy) -- for use with driver.guided_reverse_loop.

NOT YET PORTED. Reference implementation: preliminary/fisher_rao_guidance_so(n).ipynb,
the cell defining `soft_posterior_bb` / `fisher_rao_energy_bb`. Idea: replace
basic_fr's fixed uniform prior (implicitly 1/N over the k retrieved atoms) with
B draws from a Dirichlet(dirichlet_alpha) bootstrap over atom weights, solve the
same Gibbs-form posterior under each draw, and average (q_bar replaces q in
basic_fr.fisher_rao_energy). dirichlet_alpha=1.0 (flat Dirichlet) is the
canonical bootstrap; q_bar -> q only as dirichlet_alpha -> infinity.
"""
import torch


def soft_posterior_bb(y, atoms, beta_t, B=None, dirichlet_alpha=None, return_var=False):
    pass


def fisher_rao_energy_bb(y, atoms, beta_t, B=None, dirichlet_alpha=None):
    pass


def make_bayesian_fr_energy_fn(index, k, beta_t_fn, B=None, dirichlet_alpha=None):
    pass
