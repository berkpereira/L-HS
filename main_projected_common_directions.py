import problems.test_problems
from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import plotting
import matplotlib.pyplot as plt
import os

os.system('clear')

# For reference:
test_problems_list = ['rosenbrock',                      # 0
                    'powell',                            # 1
                    'well_conditioned_convex_quadratic', # 2
                    'ill_conditioned_convex_quadratic']  # 3

subspace_methods_list = ['grads',                        # 0
                         'iterates_grads',               # 1
                         'iterates_grads_diagnewtons']   # 2

# SELECT PROBLEM
PROBLEM_NAME = test_problems_list[0]
INPUT_DIM = 40
x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

# SOLVER CONFIG
SUBSPACE_METHOD = subspace_methods_list[1]

SUBSPACE_DIM = 12

REG_LAMBDA = 0.01
USE_HESS = True

RANDOM_PROJ = False

# NOTE: makes no difference in the randomised case
REPROJECT_GRAD = True

ENSEMBLE = 'haar'

ALPHA = 0.01
T_INIT = 1
TAU = 0.5

MAX_ITER = 10_000
TOL = 1e-3
ITER_PRINT_GAP = 50

# Run solver(s)
output_list = []
for append_rand_dirs in [0, 2, 4, 6, 8]:
    SOLVER_CONFIG = ProjectedCommonDirectionsConfig(obj=obj,
                                            subspace_update_method=SUBSPACE_METHOD,
                                            subspace_dim=SUBSPACE_DIM,
                                            append_rand_dirs=append_rand_dirs,
                                            reg_lambda=REG_LAMBDA,
                                            use_hess=USE_HESS,
                                            random_proj=RANDOM_PROJ,
                                            reproject_grad=REPROJECT_GRAD,
                                            ensemble=ENSEMBLE,
                                            alpha=ALPHA,
                                            t_init=T_INIT,
                                            tau=TAU,
                                            tol=TOL,
                                            max_iter=MAX_ITER,
                                            iter_print_gap=ITER_PRINT_GAP,
                                            verbose=True)

    optimiser = ProjectedCommonDirections(config=SOLVER_CONFIG)
    output = optimiser.optimise(x0)
    output_list.append(output)

# PLOTTING
plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list)
plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['update_norms'],
                                           log_plot=True)

plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['direction_norms'],
                                           log_plot=True)

plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['full_grad_norms',
                                                       'proj_grad_norms'],
                                           log_plot=True)

plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['angles_to_full_grad'],
                                           log_plot=False)

# plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
#                                            attr_names=['cond_nos'],
#                                            log_plot=True)

# plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
#                                            attr_names=['P_ranks'],
#                                            log_plot=False)

# plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
#                                                 attr_names=['f_vals', 'P_ranks'],
#                                                 log_plots=[True, False])

# plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
#                                                 attr_names=['f_vals', 'angles_to_full_grad'],
#                                                 log_plots=[True, False])

plt.show()