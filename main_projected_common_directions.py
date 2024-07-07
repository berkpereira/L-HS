import problems.test_problems
import autograd.numpy as np
from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import plotting
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.system('clear')

################################################################################
################################################################################
################################################################################

# For reference:
test_problems_list = ['rosenbrock',                      # 0
                    'powell',                            # 1
                    'well_conditioned_convex_quadratic', # 2
                    'ill_conditioned_convex_quadratic']  # 3

# SELECT PROBLEM
PROBLEM_NAME = test_problems_list[0]
INPUT_DIM = 12
x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

# SOLVER CONFIG

REG_LAMBDA = 0.01
USE_HESS = True

INNER_USE_FULL_GRAD = True
REPROJECT_GRAD = False       # NOTE: makes no difference in the randomised proj variant

RANDOM_PROJ = True

ENSEMBLE = 'haar'
ALPHA = 0.01
T_INIT = 1
TAU = 0.5

MAX_ITER = np.inf
DERIV_BUDGET = 2_000

TOL = 1e-4
ITER_PRINT_GAP = 20

# Run solver(s)
output_list = []

SUBSPACE_NO_LIST = [(3, 0, 0),
                    (1, 2, 0),
                    (2, 1, 0),
                    (1, 1, 1)]

for SUBSPACE_NO_GRADS, SUBSPACE_NO_UPDATES, SUBSPACE_NO_RANDOM in SUBSPACE_NO_LIST:
    SUBSPACE_DIM = SUBSPACE_NO_GRADS + SUBSPACE_NO_UPDATES + SUBSPACE_NO_RANDOM
    RANDOM_PROJ_DIM = SUBSPACE_DIM # Does not need to be...
    for _ in range(2): # how many runs per solver?
        SOLVER_CONFIG = ProjectedCommonDirectionsConfig(obj=obj,
                                                        subspace_no_grads=SUBSPACE_NO_GRADS,
                                                        subspace_no_updates=SUBSPACE_NO_UPDATES,
                                                        subspace_no_random=SUBSPACE_NO_RANDOM,
                                                        inner_use_full_grad=INNER_USE_FULL_GRAD,
                                                        reg_lambda=REG_LAMBDA,
                                                        use_hess=USE_HESS,
                                                        random_proj=RANDOM_PROJ,
                                                        random_proj_dim=RANDOM_PROJ_DIM,
                                                        reproject_grad=REPROJECT_GRAD,
                                                        ensemble=ENSEMBLE,
                                                        alpha=ALPHA,
                                                        t_init=T_INIT,
                                                        tau=TAU,
                                                        tol=TOL,
                                                        max_iter=MAX_ITER,
                                                        deriv_budget=DERIV_BUDGET,
                                                        iter_print_gap=ITER_PRINT_GAP,
                                                        verbose=True)

        optimiser = ProjectedCommonDirections(config=SOLVER_CONFIG)
        output = optimiser.optimise(x0)
        output_list.append(output)

# PLOTTING
plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list,
                                         deriv_evals_axis=True)

plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['update_norms'],
                                           log_plot=True)

# plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
#                                            attr_names=['direction_norms'],
#                                            log_plot=True)

plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['full_grad_norms',
                                                       'proj_grad_norms'],
                                           log_plot=True)

# plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
#                                            attr_names=['angles_to_full_grad'],
#                                            log_plot=False)

# plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
#                                            attr_names=['cond_nos'],
#                                            log_plot=True)

# plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
#                                            attr_names=['P_ranks'],
#                                            log_plot=False)

# plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
#                                            attr_names=['P_norms'],
#                                            log_plot=True)

plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
                                                attr_names=['f_vals', 'P_ranks'],
                                                log_plots=[True, False])

plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
                                                attr_names=['proj_grad_norms', 'P_ranks'],
                                                log_plots=[True, False])

# plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
#                                                 attr_names=['f_vals', 'P_norms'],
#                                                 log_plots=[True, True])

plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
                                                attr_names=['P_norms', 'P_ranks'],
                                                log_plots=[True, False])

# plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
#                                                 attr_names=['update_norms', 'P_ranks'],
#                                                 log_plots=[True, False])

# plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
#                                                 attr_names=['f_vals', 'angles_to_full_grad'],
#                                                 log_plots=[True, False])

plt.show()