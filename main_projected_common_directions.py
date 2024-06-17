import problems.test_problems
from solvers.subspaces.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import plotting
import os
os.system('clear')

# For reference:
test_problems_list = ['rosenbrock', # 0
                    'powell', # 1
                    'well_conditioned_convex_quadratic', # 2
                    'ill_conditioned_convex_quadratic'] # 3

subspace_methods_list = ['grads', # 0
                         'iterates_grads', # 1
                         'iterates_grads_diagnewtons'] # 2

# SELECT PROBLEM
PROBLEM_NAME = test_problems_list[2]
INPUT_DIM = 10
x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

# SOLVER CONFIG
SUBSPACE_METHOD = subspace_methods_list[0]
REG_LAMBDA = 0.001
USE_HESS = True

RANDOM_PROJ = False

ENSEMBLE = 'scaled_gaussian'
ALPHA = 0.01
T_INIT = 1
TAU = 0.5

MAX_ITER = 3000
TOL = 1e-3
ITER_PRINT_GAP = 1

# Run solver(s)
output_list = []

SUBSPACE_DIM = 7

SOLVER_CONFIG = ProjectedCommonDirectionsConfig(obj=obj,
                                        subspace_update_method=SUBSPACE_METHOD,
                                        subspace_dim=SUBSPACE_DIM,
                                        reg_lambda=REG_LAMBDA,
                                        use_hess=USE_HESS,
                                        random_proj=RANDOM_PROJ,
                                        ensemble=ENSEMBLE,
                                        alpha=ALPHA,
                                        t_init=T_INIT,
                                        tau=TAU,
                                        tol=TOL,
                                        max_iter=MAX_ITER,
                                        iter_print_gap=ITER_PRINT_GAP,
                                        verbose=True)

print(f'INPUT DIMENSION: {INPUT_DIM}. SOLVER WITH SUBSPACE DIMENSION: {SUBSPACE_DIM}')

optimiser = ProjectedCommonDirections(config=SOLVER_CONFIG)
output = optimiser.optimise(x0)
output_list.append(output)

# Plot
plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list)
