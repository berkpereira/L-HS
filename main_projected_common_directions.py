import problems.test_problems
from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import plotting
import matplotlib.pyplot as plt
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
PROBLEM_NAME = test_problems_list[0]
INPUT_DIM = 20
x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

# SOLVER CONFIG
SUBSPACE_METHOD = subspace_methods_list[0]
REG_LAMBDA = 0.01
USE_HESS = True

RANDOM_PROJ = True

ENSEMBLE = 'scaled_gaussian'
ALPHA = 0.01
T_INIT = 1
TAU = 0.5

MAX_ITER = 100
TOL = 1e-3
ITER_PRINT_GAP = 1

# Run solver(s)
output_list = []

SUBSPACE_DIM = 6

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

# PLOTTING
plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list)
plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['update_norms'],
                                           log_plot=True)

plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['full_grad_norms', 'proj_grad_norms'],
                                           log_plot=True)

plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['angles_to_full_grad'],
                                           log_plot=False)
plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['cond_nos'],
                                           log_plot=True)

plt.show()