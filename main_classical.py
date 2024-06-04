import problems.test_problems
import solvers
import plotting

# For reference:
test_problems_list = ['rosenbrock', # 0
                    'powell', # 1
                    'well_conditioned_convex_quadratic', # 2
                    'ill_conditioned_convex_quadratic'] # 3

method_list = ['SD',
                'Newton']

PROBLEM_NAME = test_problems_list[1]
METHOD = 'Newton'
INPUT_DIM = 40 # NOTE: depending on the problem, this may have no effect

ITER_PRINT_GAP = 1

# Instantiate problem
x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

# Initialise optimiser
optimiser = solvers.classical.linesearch_bArmijo.LinesearchBacktrackingArmijo(method=METHOD, obj=obj, alpha=0.01, t_init=1, tol = 1e-4, max_iter=10000, iter_print_gap=ITER_PRINT_GAP, verbose=True)

# Perform optimisation
output = optimiser.optimise(x0)

plotting.plotting.plot_loss_vs_iteration(f_vals=output.f_vals)
