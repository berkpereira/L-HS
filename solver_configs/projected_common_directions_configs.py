"""
This file is meant to store a curated set of configurations of
ProjectedCommonDirections 
"""
import autograd.numpy as np
from solvers.projected_common_directions import ProjectedCommonDirectionsConfig

# Dictionary to hold configuration parameters for all solver variants
solver_variants_dict = {
    "solver0": {
        "subspace_frac_grads": 0,
        "subspace_frac_updates": 0,
        "subspace_frac_random": 1,
        "random_proj_dim": 5,
        "direction_str": 'newton',
        "reg_lambda": 0.01,
        "ensemble": 'haar',
        "orth_P_k": True,
        "alpha": 0.01,
        "t_init": 1,
        "tau": 0.5
    },
    "solver1": {
        "subspace_frac_grads": 0,
        "subspace_frac_updates": 0,
        "subspace_frac_random": 0.3,
        "random_proj_dim_frac": 0,
        "direction_str": 'sd',
        "reg_lambda": 0.01,
        "ensemble": 'haar',
        "orth_P_k": True,
        "alpha": 0.01,
        "t_init": 100,
        "tau": 0.5
    },
    "solver2": {
        "subspace_frac_grads": 0.1,
        "subspace_frac_updates": 0.1,
        "subspace_frac_random": 0.1,
        "random_proj_dim_frac": 0.1,
        "direction_str": 'sd',
        "reg_lambda": 0.01,
        "ensemble": 'haar',
        "orth_P_k": True,
        "alpha": 0.01,
        "t_init": 100,
        "tau": 0.5
    },
    "solver3": {
        "subspace_frac_grads": 0.1,
        "subspace_frac_updates": 0.1,
        "subspace_frac_random": 0.1,
        "random_proj_dim_frac": 0.2,
        "direction_str": 'sd',
        "reg_lambda": 0.01,
        "ensemble": 'haar',
        "orth_P_k": True,
        "alpha": 0.01,
        "t_init": 100,
        "tau": 0.5
    },
    "solver4": {
        "subspace_frac_grads": 0.1,
        "subspace_frac_updates": 0.1,
        "subspace_frac_random": 0.1,
        "random_proj_dim_frac": 0.2,
        "direction_str": 'newton',
        "reg_lambda": 0.01,
        "ensemble": 'haar',
        "orth_P_k": True,
        "alpha": 0.01,
        "t_init": 100,
        "tau": 0.5
    },
    # Add more variants as needed
}

# Function to create a configuration object for a given variant name
def create_config(obj, variant_name):
    variant = solver_variants_dict[variant_name]
    return ProjectedCommonDirectionsConfig(obj=obj, **variant)