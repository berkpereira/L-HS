import problems.test_problems
from solvers.cartis2022algorithm3 import Cartis2022Algorithm3, Cartis2022Algorithm3Config
import os
import matplotlib.pyplot as plt
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
INPUT_DIM = 20

SUBSPACE_DIM = 5

TOL = 1e-4
OUTER_MAX_ITER = 40_000
INNER_MAX_ITER = 10

ITER_PRINT_GAP = 100

GAMMA1 = 0.5
CONST_C = 1
CONST_P = 6
KAPPA_T = 0.1
THETA = 0.8
ALPHA_MAX = 1
ENSEMBLE = 'scaled_gaussian'

def main():
    # Instantiate problem
    x0, obj = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)

    output_list = []

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

    plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                               attr_names=['successful_iters'],
                                               use_markers=True,
                                               marker_str='x',
                                               log_plot=False)
    
    plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                               attr_names=['update_norms'],
                                               use_markers=False,
                                               log_plot=True)
    
    plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                               attr_names=['full_grad_norms',
                                                           'proj_grad_norms',
                                                           'red_grad_norms'],
                                                           alpha=0.5,
                                                           log_plot=True)
    
    plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
                                               attr_names=['proj_full_grad_norm_ratios',
                                                           'red_full_grad_norm_ratios'],
                                                           alpha=0.5,
                                                           log_plot=True)

    plt.show()

# May the main function
if __name__ == '__main__':
    main()