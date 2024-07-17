"""
This file complements solver-proper configurations by storing a library
of so-called 'passable' configurations to be used --- this relates to quantities such
as the maximum number of iterations or the maximum number of derivative evaluations,
the tolerance to termination... things like that.
This is essentially relative to 'passable attributes' (apart from
the problem (obj object) itself, which we forgo here).
"""
import autograd.numpy as np

passable_variants_dict = {
    "passable0": {
        "tol": 1e-6,
        "max_iter": np.inf,
        "deriv_budget": 5_000,
        "iter_print_gap": 50,
        "verbose": True
    },
    "passable1": {
        "tol": 1e-4,
        "max_iter": np.inf,
        "deriv_budget": 10_000,
        "iter_print_gap": 50,
        "verbose": True
    },
    "passable2": {
        "tol": 1e-4,
        "max_iter": np.inf,
        "equiv_grad_budget": 500,
        "iter_print_gap": 50,
        "verbose": True
    },
    # Add more variants as needed
}