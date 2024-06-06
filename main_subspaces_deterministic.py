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

subspace_methods_list = ['grads',
                         'iterates_grads',
                         'iterates_grads_diagnewtons']

PROBLEM_NAME = test_problems_list[1]
INPUT_DIM = 24 # NOTE: depending on the problem, this may have no effect
SUBSPACE_DIM = 12
SUBSPACE_METHOD = subspace_methods_list[1]
REG_LAMBDA = 0.01

ITER_PRINT_GAP = 20

TOL = 1e-5
MAX_ITER = 10000

# Instantiate problem
x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

# Initialise optimiser
optimiser = solvers.subspaces_deterministic.common_directions.CommonDirections(obj=obj, subspace_update_method=SUBSPACE_METHOD,subspace_dim=SUBSPACE_DIM, reg_lambda=REG_LAMBDA, alpha=0.01, t_init=1, tol=TOL, max_iter=MAX_ITER, iter_print_gap=ITER_PRINT_GAP, verbose=True)

# Perform optimisation
output_list = []
for input_dim in range(6, 25, 6):
    output = optimiser.optimise(x0)
    output_list.append(output)

plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list)
