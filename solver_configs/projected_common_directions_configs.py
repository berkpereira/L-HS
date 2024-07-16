"""
This file is meant to store a curated set of configurations of
ProjectedCommonDirections 
"""
import autograd.numpy as np
from solvers.projected_common_directions import ProjectedCommonDirectionsConfig

# Dictionary to hold configuration parameters for all solver variants
solver_variants_dict = {
    "solver0": {
        "subspace_no_grads": 1,
        "subspace_no_updates": 1,
        "subspace_no_random": 2,
        "direction_str": 'sd',
        "reg_lambda": 0.01,
        "use_hess": True,
        "random_proj": True,
        "random_proj_dim": 5,
        "reproject_grad": False,
        "ensemble": 'haar',
        "inner_use_full_grad": True,
        "alpha": 0.01,
        "t_init": 100,
        "tau": 0.5
    },
    "solver1": {
        "subspace_no_grads": 2,
        "subspace_no_updates": 1,
        "subspace_no_random": 2,
        "direction_str": 'sd',
        "reg_lambda": 0.01,
        "use_hess": True,
        "random_proj": True,
        "random_proj_dim": 5,
        "reproject_grad": False,
        "ensemble": 'haar',
        "inner_use_full_grad": True,
        "alpha": 0.01,
        "t_init": 100,
        "tau": 0.5
    },
    "solver2": {
        "subspace_frac_grads": 0.1,
        "subspace_frac_updates": 0.1,
        "subspace_frac_random": 0.1,
        "direction_str": 'sd',
        "reg_lambda": 0.01,
        "use_hess": True,
        "random_proj": True,
        "random_proj_dim": 5,
        "reproject_grad": False,
        "ensemble": 'haar',
        "inner_use_full_grad": True,
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