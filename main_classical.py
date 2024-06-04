import problems.test_problems
import solvers.classical.linesearch_bArmijo
import plotting.plotting

# For reference:
test_problems_list = ['rosenbrock',
                    'powell_singular',
                    'well_conditioned_convex_quadratic',
                    'ill_conditioned_convex_quadratic']

method_list = ['SD',
                'Newton']

METHOD = 'Newton'
PROBLEM_NAME = 'rosenbrock'
INPUT_DIM = 20 # NOTE: depending on the problem, this may have no effect

# Instantiate problem
x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

# Initialize optimiser
optimiser = solvers.classical.linesearch_bArmijo.LinesearchBacktrackingArmijo(method=METHOD, obj=obj, alpha=0.01, t_init=1, tol = 1e-6, max_iter=10000, verbose=True)

print(x0)

# Perform optimisation
optimal_x, f_vals = optimiser.optimise(x0)

plotting.plotting.plot_loss_vs_iteration(f_vals=f_vals)
