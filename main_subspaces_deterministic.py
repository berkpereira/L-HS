import problems.test_problems
import solvers
import plotting

# For reference:
test_problems_list = ['rosenbrock', # 0
                    'powell', # 1
                    'well_conditioned_convex_quadratic', # 2
                    'ill_conditioned_convex_quadratic'] # 3

PROBLEM_NAME = test_problems_list[0]
INPUT_DIM = 20 # NOTE: depending on the problem, this may have no effect
SUBSPACE_DIM = 18
REG_LAMBDA = 0.01

ITER_PRINT_GAP = 20

# Instantiate problem
x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

# Initialise optimiser
optimiser = solvers.subspaces_deterministic.common_directions.CommonDirections(obj=obj, subspace_dim=SUBSPACE_DIM, reg_lambda = 0.01, alpha=0.01, t_init=1, tol = 1e-4, max_iter=10000, iter_print_gap=ITER_PRINT_GAP, verbose=True)

# Perform optimisation
output = optimiser.optimise(x0)

plotting.plotting.plot_loss_vs_iteration(f_vals=output.f_vals)
