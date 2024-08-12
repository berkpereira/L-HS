"""
This file is meant to store a curated set of configurations for use in 
numerical experiments for thesis results.
"""
import autograd.numpy as np
from solvers.projected_common_directions import ProjectedCommonDirectionsConfig

# Dictionary to hold configuration parameters for all solver variants
solver_config_tree = {
    "sd": { # 
        'orth_Pk': { # orthogonalise Pk ?
            'solver1': {
                "subspace_frac_grads": 0.1,
                "subspace_frac_updates": 0.1,
                "subspace_frac_random": 0.1,
                "random_proj_dim_frac": 1,
                "direction_str": 'sd',
                "ensemble": 'haar',
                "orth_P_k": True,
                "normalise_P_k_cols": False,
                "alpha_max": 100,
            },
            'solver2': {
                "subspace_frac_grads": 0.1,
                "subspace_frac_updates": 0.1,
                "subspace_frac_random": 0.1,
                "random_proj_dim_frac": 1,
                "direction_str": 'sd',
                "ensemble": 'haar',
                "orth_P_k": False,
                "normalise_P_k_cols": True,
                "alpha_max": 100,
            },
            'solver3': {
                "subspace_frac_grads": 0.1,
                "subspace_frac_updates": 0.1,
                "subspace_frac_random": 0.1,
                "random_proj_dim_frac": 0.2,
                "direction_str": 'sd',
                "ensemble": 'haar',
                "orth_P_k": True,
                "normalise_P_k_cols": False,
                "alpha_max": 100,
            },
            'solver4': {
                "subspace_frac_grads": 0.1,
                "subspace_frac_updates": 0.1,
                "subspace_frac_random": 0.1,
                "random_proj_dim_frac": 0.2,
                "direction_str": 'sd',
                "ensemble": 'haar',
                "orth_P_k": False,
                "normalise_P_k_cols": True,
                "alpha_max": 100,
            }
        }
    },
    "newton": {
        'orth_Pk'
    },
    
################################################################################
    
    "sample_solvers": {
        "solver0": {
            "subspace_frac_grads": 0.1,
            "subspace_frac_updates": 0.1,
            "subspace_frac_random": 0.1,
            "random_proj_dim_frac": 0.2,
            "direction_str": 'sd',
            "reg_lambda": 0.01,
            "ensemble": 'haar',
            "orth_P_k": True,
            "normalise_P_k_cols": False,
            "alpha_max": 100,
            "tau": 0.5,
            "c_const": 1,
            "N_try": 100,
        },
        "solver1": {
            "subspace_frac_grads": 0.1,
            "subspace_frac_updates": 0.1,
            "subspace_frac_random": 0.1,
            "random_proj_dim_frac": 0.2,
            "direction_str": 'sd',
            "reg_lambda": 0.01,
            "ensemble": 'haar',
            "orth_P_k": False,
            "normalise_P_k_cols": True,
            "alpha_max": 100,
            "tau": 0.5,
            "c_const": 1,
            "N_try": 100,
        },
        "solver2": {
            "subspace_frac_grads": 0.1,
            "subspace_frac_updates": 0.1,
            "subspace_frac_random": 0.1,
            "random_proj_dim_frac": 0.2,
            "direction_str": 'sd',
            "reg_lambda": 0.01,
            "ensemble": 'haar',
            "orth_P_k": True,
            "normalise_P_k_cols": False,
            "alpha_max": 100,
            "tau": 0.5,
            "c_const": 1,
            "N_try": 100,
        },
        "solver3": {
            "subspace_frac_grads": 0.1,
            "subspace_frac_updates": 0.1,
            "subspace_frac_random": 0.1,
            "random_proj_dim_frac": 0.4,
            "direction_str": 'sd',
            "reg_lambda": 0.01,
            "ensemble": 'haar',
            "orth_P_k": False,
            "normalise_P_k_cols": True,
            "alpha_max": 100,
            "tau": 0.5,
            "c_const": 1,
            "N_try": 100,
        },
        "solver4": {
            "subspace_frac_grads": 0.1,
            "subspace_frac_updates": 0.1,
            "subspace_frac_random": 0.1,
            "random_proj_dim_frac": 0.2,
            "direction_str": 'sd',
            "reg_lambda": 0.01,
            "ensemble": 'haar',
            "orth_P_k": True,
            "normalise_P_k_cols": False,
            "alpha_max": 100,
            "tau": 0.5,
            "c_const": 1,
            "N_try": 100,
        },
        "solver5": {
            "subspace_frac_grads": 0.1,
            "subspace_frac_updates": 0.1,
            "subspace_frac_random": 0.1,
            "random_proj_dim_frac": 0.2,
            "direction_str": 'sd',
            "reg_lambda": 0.01,
            "ensemble": 'haar',
            "orth_P_k": True,
            "normalise_P_k_cols": False,
            "alpha_max": 100,
            "tau": 0.5,
            "c_const": 100,
            "N_try": 100,
        },
    },
    "solve_best": {
        "subspace_frac_grads": 0,
        "subspace_frac_updates": 0,
        "subspace_frac_random": 1,
        "random_proj_dim_frac": 0,
        "direction_str": 'newton',
        "reg_lambda": 0.01,
        "ensemble": 'haar',
        "orth_P_k": True,
        "normalise_P_k_cols": False,
        "alpha_max": 1,
        "tau": 0.5,
        "c_const": np.inf,
        "N_try": np.inf,
    },
    "full_sd": {
        "subspace_frac_grads": 0,
        "subspace_frac_updates": 0,
        "subspace_frac_random": 1,
        "random_proj_dim_frac": 0,
        "direction_str": 'sd',
        "reg_lambda": 0.01,
        "ensemble": 'haar',
        "orth_P_k": True,
        "normalise_P_k_cols": False,
        "alpha_max": 1,
        "tau": 0.5,
        "c_const": np.inf,
        "N_try": np.inf,
    },
    "full_newton": {
        "subspace_frac_grads": 0,
        "subspace_frac_updates": 0,
        "subspace_frac_random": 1,
        "random_proj_dim_frac": 0,
        "direction_str": 'newton',
        "reg_lambda": 0.01,
        "ensemble": 'haar',
        "orth_P_k": True,
        "normalise_P_k_cols": False,
        "alpha_max": 1,
        "tau": 0.5,
        "c_const": np.inf,
        "N_try": np.inf,
    },
    "lee2022_grads": {
        "subspace_frac_grads": 0.2,
        "subspace_frac_updates": 0,
        "subspace_frac_random": 0,
        "random_proj_dim_frac": 1,
        "direction_str": 'newton',
        "reg_lambda": 0.01,
        "ensemble": 'haar',
        "orth_P_k": True,
        "normalise_P_k_cols": False,
        "alpha_max": 1,
        "tau": 0.5,
        "c_const": np.inf,
        "N_try": np.inf,
    },
    "lee2022_grads_updates": {
        "subspace_frac_grads": 0.1,
        "subspace_frac_updates": 0.1,
        "subspace_frac_random": 0,
        "random_proj_dim_frac": 1,
        "direction_str": 'newton',
        "reg_lambda": 0.01,
        "ensemble": 'haar',
        "orth_P_k": True,
        "normalise_P_k_cols": False,
        "alpha_max": 1,
        "tau": 0.5,
        "c_const": np.inf,
        "N_try": np.inf,
    },
    "random_subspace": {
        "subspace_frac_grads": 0,
        "subspace_frac_updates": 0,
        "subspace_frac_random": 0.2,
        "random_proj_dim_frac": 0,
        "direction_str": 'sd',
        "reg_lambda": 0.01,
        "ensemble": 'haar',
        "orth_P_k": False,
        "normalise_P_k_cols": False,
        "alpha_max": 100,
        "tau": 0.5,
        "c_const": np.inf,
        "N_try": np.inf,
    },
}

# Function to create a configuration object for a given variant name
def create_config(obj, variant_name):
    variant = solver_variants_dict[variant_name]
    return ProjectedCommonDirectionsConfig(obj=obj, **variant)