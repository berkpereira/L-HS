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

PROBLEM_NAME = test_problems_list[0]
METHOD = 'SD'
INPUT_DIM = 20 # NOTE: depending on the problem, this may have no effect

ITER_PRINT_GAP = 20
TOL = 1e-6

# Instantiate problem
x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

# Initialise optimiser
optimiser = solvers.classical.linesearch_bArmijo.LinesearchBacktrackingArmijo(method=METHOD, obj=obj, alpha=0.01, t_init=1, tol = TOL, max_iter=10000, iter_print_gap=ITER_PRINT_GAP, verbose=True)

# Perform optimisation
output = optimiser.optimise(x0)
output_list = [output]

plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list)
