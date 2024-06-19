import problems.test_problems
from solvers.BFGS import BFGSLinesearchConfig, BFGSLinesearch
import plotting
import matplotlib.pyplot as plt
import autograd.numpy as np
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
INPUT_DIM = 30
x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

# Initial inverse Hessian approximation
H0 = np.identity(INPUT_DIM)

# SOLVER CONFIG
C1 = 0.0001 # Armijo condition scaling in linesearch
C2 = 0.9 # Strong curvature condition scaling in linesearch
LINESEARCH_MAX_ITER = 100 # Maximum number of inner iterations in the linesearch procedure.

TOL = 1e-6
MAX_ITER = 1_000
ITER_PRINT_GAP = 20

SOLVER_CONFIG = BFGSLinesearchConfig(obj=obj,
                                     c1=C1,
                                     c2=C2,
                                     linesearch_max_iter=LINESEARCH_MAX_ITER,
                                     tol=TOL,
                                     max_iter=MAX_ITER,
                                     iter_print_gap=ITER_PRINT_GAP,
                                     verbose=True)

# Run solver(s)
output_list = []
optimiser = BFGSLinesearch(config=SOLVER_CONFIG)
output = optimiser.optimise(x0, H0)
output_list.append(output)

# Plot stuff
plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list)
plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['update_norms'],
                                           log_plot=True)
plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['angles_to_grad'],
                                           log_plot=False)
plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                           attr_names=['grad_norms'],
                                           log_plot=True)

plt.show()