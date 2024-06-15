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

subspace_methods_list = ['grads', # 0
                         'iterates_grads', # 1
                         'iterates_grads_diagnewtons'] # 2

PROBLEM_NAME = test_problems_list[0]
INPUT_DIM = 24 # NOTE: depending on the problem, this may have no effect

SUBSPACE_METHOD = subspace_methods_list[0]
REG_LAMBDA = 0.001

TOL = 1e-10
MAX_ITER = 100

ITER_PRINT_GAP = 20

# Instantiate problem
x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

# Run solver(s)
output_list = []
for SUBSPACE_DIM in range(6, 25, 6):
    print(f'INPUT DIMENSION: {INPUT_DIM}. SOLVER WITH SUBSPACE DIMENSION: {SUBSPACE_DIM}')
    optimiser = solvers.subspaces_deterministic.common_directions.CommonDirections(obj=obj, subspace_update_method=SUBSPACE_METHOD,subspace_dim=SUBSPACE_DIM, reg_lambda=REG_LAMBDA, alpha=0.01, t_init=1, tol=TOL, max_iter=MAX_ITER, iter_print_gap=ITER_PRINT_GAP, verbose=False)    
    output = optimiser.optimise(x0)
    output_list.append(output)

# Plot
plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list)
