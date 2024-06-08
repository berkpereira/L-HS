import problems.test_problems
import solvers
import plotting
import os
import cProfile
import pstats
import io

os.system('clear')

# For reference:
test_problems_list = ['rosenbrock',                      # 0
                    'powell',                            # 1
                    'well_conditioned_convex_quadratic', # 2
                    'ill_conditioned_convex_quadratic' ] # 3


PROBLEM_NAME = test_problems_list[2]
INPUT_DIM = 100 # NOTE: depending on the problem, this may have no effect

SUBSPACE_DIM = 10

TOL = 0 # for algorithm run up to outer max iters
OUTER_MAX_ITER = 3000
INNER_MAX_ITER = 10

ITER_PRINT_GAP = 1

GAMMA1 = 0.5
CONST_C = 1
CONST_P = 1
KAPPA_T = 0.1
THETA = 0.8
ALPHA_MAX = 1
ENSEMBLE = 'scaled_gaussian'

def main():
    # Instantiate problem
    x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

    # Run solver(s)
    optimiser = solvers.subspaces_random.cartis2022algorithm3.Cartis2022Algorithm3(obj, subspace_dim=SUBSPACE_DIM, gamma1=GAMMA1, const_c=CONST_C, const_p=CONST_P, kappa_T=KAPPA_T, theta=THETA, alpha_max=ALPHA_MAX, ensemble=ENSEMBLE, tol=TOL, inner_max_iter=INNER_MAX_ITER, outer_max_iter=OUTER_MAX_ITER, iter_print_gap=ITER_PRINT_GAP, verbose=True)

    output_list = [optimiser.optimise(x0)]

    # Plot
    # plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list)

# Profile the main function
if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()

    if not input('CHANGE THE FILE NAME, YOU NUGGET (ok): ') == 'ok':
        raise Exception('Nugget')
    
    results_file_name = 'profiler_results_full_grad.txt'

    # Write profiling results to a text file
    with open(results_file_name, 'w') as f:
        ps = pstats.Stats(pr, stream=f).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()

    print(f'Profiling results have been written to {results_file_name}')