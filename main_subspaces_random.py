import problems.test_problems
import solvers
import plotting
import os
os.system('clear')

# For reference:
test_problems_list = ['rosenbrock', # 0
                    'powell', # 1
                    'well_conditioned_convex_quadratic', # 2
                    'ill_conditioned_convex_quadratic'] # 3


PROBLEM_NAME = test_problems_list[0]
INPUT_DIM = 20 # NOTE: depending on the problem, this may have no effect

TOL = 1e-6
OUTER_MAX_ITER = 10000
INNER_MAX_ITER = 100

ITER_PRINT_GAP = 20

GAMMA1 = 0.5
CONST_C = 1
CONST_P = 1
KAPPA_T = 0.1
THETA = 0.8
ALPHA_MAX = 1
ENSEMBLE = 'whatever'

# Instantiate problem
x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

# Run solver(s)
optimiser = solvers.subspaces_random.cartis2022algorithm3.Cartis2022Algorithm3(obj, subspace_dim=INPUT_DIM, gamma1=GAMMA1, const_c=CONST_C, const_p=CONST_P, kappa_T=KAPPA_T, theta=THETA, alpha_max=ALPHA_MAX, ensemble=ENSEMBLE, tol=TOL, inner_max_iter=INNER_MAX_ITER, outer_max_iter=OUTER_MAX_ITER, iter_print_gap=ITER_PRINT_GAP, verbose=True)

output_list = [optimiser.optimise(x0)]


# Plot
plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list)
