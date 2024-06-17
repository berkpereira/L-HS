import problems.test_problems
from solvers.subspaces.cartis2022algorithm3 import Cartis2022Algorithm3, Cartis2022Algorithm3Config
import os
import plotting
import cProfile
import pstats

os.system('clear')

# For reference:
test_problems_list = ['rosenbrock',                      # 0
                    'powell',                            # 1
                    'well_conditioned_convex_quadratic', # 2
                    'ill_conditioned_convex_quadratic' ] # 3


PROBLEM_NAME = test_problems_list[0]
INPUT_DIM = 32 # NOTE: depending on the problem, this may have no effect

TOL = 1e-2
OUTER_MAX_ITER = 20000
INNER_MAX_ITER = 10

ITER_PRINT_GAP = 100

GAMMA1 = 0.5
CONST_C = 1
CONST_P = 6
KAPPA_T = 0.1
THETA = 0.5
ALPHA_MAX = 1
ENSEMBLE = 'scaled_gaussian'

def main():
    # Instantiate problem
    x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

    output_list = []
    for SUBSPACE_DIM in range(14, 15):
        SOLVER_CONFIG = Cartis2022Algorithm3Config(obj,
                                                subspace_dim=SUBSPACE_DIM,
                                                gamma1=GAMMA1,
                                                const_c=CONST_C,
                                                const_p=CONST_P,
                                                kappa_T=KAPPA_T,
                                                theta=THETA,
                                                alpha_max=ALPHA_MAX,
                                                ensemble=ENSEMBLE,
                                                tol=TOL,
                                                inner_max_iter=INNER_MAX_ITER,
                                                outer_max_iter=OUTER_MAX_ITER,
                                                iter_print_gap=ITER_PRINT_GAP,
                                                verbose=True)
        
        print(f'INPUT DIMENSION: {INPUT_DIM}. SOLVER WITH SUBSPACE DIMENSION: {SUBSPACE_DIM}')
        
        optimiser = Cartis2022Algorithm3(config=SOLVER_CONFIG)
        output = optimiser.optimise(x0=x0)
        output_list.append(output)

    # Plot
    plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list)

# May the main function
if __name__ == '__main__':
    main()