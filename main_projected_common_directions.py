import problems.test_problems
import autograd.numpy as np
from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
from solvers.utils import average_solver_runs
import plotting
import matplotlib.pyplot as plt
import os






# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# TODO:
# CURRENTLY REFACTORING MAIN SCRIPT CODE. SEE BROWSER FOR PROGRESS AND RELEVANT STUFF.











def set_seed(seed):
    np.random.seed(42)

def get_problem(problem_name, input_dim):
    return problems.test_problems.select_problem(problem_name=problem_name, input_dim=input_dim)

def configure_solver(obj, subspace_no_grads, subspace_no_updates, subspace_no_random, **kwargs):
    subspace_dim = subspace_no_grads + subspace_no_updates + subspace_no_random
    random_proj_dim = subspace_dim  # This NEED NOT be the case...
    config = ProjectedCommonDirectionsConfig(
        obj=obj,
        subspace_no_grads=subspace_no_grads,
        subspace_no_updates=subspace_no_updates,
        subspace_no_random=subspace_no_random,
        random_proj_dim=random_proj_dim,
        **kwargs
    )
    return ProjectedCommonDirections(config=config)

def run_solvers(problem_tup, solvers_list, no_runs, result_attrs):
    avg_results_dict = average_solver_runs(problem_tup, solvers_list, no_runs, result_attrs)
    return avg_results_dict

def plot_run_solvers(output_list):
    # Uncomment the following lines to 'select' plotting functions
    plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list,
                                             deriv_evals_axis=True)
    plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list,
                                             deriv_evals_axis=False)
    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['update_norms'], log_plot=True)
    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['direction_norms'], log_plot=True)
    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['full_grad_norms', 'proj_grad_norms'], log_plot=True)
    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['angles_to_full_grad'], log_plot=False)
    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['cond_nos'], log_plot=True)
    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['P_ranks'], log_plot=False)
    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['P_norms'], log_plot=True)
    # plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                                 attr_names=['f_vals', 'P_ranks'], log_plots=[True, False])
    # plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                                 attr_names=['proj_grad_norms', 'P_ranks'], log_plots=[True, False])
    # plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                                 attr_names=['f_vals', 'P_norms'], log_plots=[True, True])
    # plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                                 attr_names=['P_norms', 'P_ranks'], log_plots=[True, False])
    # plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                                 attr_names=['update_norms', 'P_ranks'], log_plots=[True, False])
    # plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                                 attr_names=['f_vals', 'angles_to_full_grad'], log_plots=[True, False])
    pass

def run_average_solvers(problem_tup, solvers_list, no_runs, result_attrs):
    avg_results_dict = average_solver_runs(problem_tup, solvers_list, no_runs, result_attrs)
    return avg_results_dict

def plot_average_results(avg_results_dict, result_attrs):
    plotting.plotting.plot_solver_averages(avg_results_dict, result_attrs)

def main():
    os.system('clear')
    set_seed(42)

    test_problems_list = ['rosenbrock',
                          'powell',
                          'well_conditioned_convex_quadratic',
                          'ill_conditioned_convex_quadratic']
    problem_name = test_problems_list[3]
    input_dim = 12
    problem_tup = get_problem(problem_name, input_dim)
    x0, obj = problem_tup

    solver_config_params = {
        'reg_lambda': 0.01,
        'use_hess': True,
        'inner_use_full_grad': True,
        'reproject_grad': False,
        'direction_str': 'sd',
        'random_proj': True,
        'ensemble': 'haar',
        'alpha': 0.01,
        't_init': 1,
        'tau': 0.5,
        'tol': 1e-4,
        'max_iter': np.inf,
        'deriv_budget': 300,
        'iter_print_gap': 20,
        'verbose': True
    }

    subspace_no_list = [(0, 0, 3),
                        (1, 1, 1)]
    solvers_list = []
    output_list = []

    for subspace_no_grads, subspace_no_updates, subspace_no_random in subspace_no_list:
        solver = configure_solver(obj, subspace_no_grads, subspace_no_updates, subspace_no_random, **solver_config_params)
        solvers_list.append(solver)

        # Running each solver and collecting output
        for _ in range(4):
            output = solver.optimise(x0)
            output_list.append(output)

    avg_results_attrs = ['final_f_val']
    avg_results_dict = run_solvers(problem_tup, solvers_list, no_runs=2, result_attrs=avg_results_attrs)

    plot_average_results(avg_results_dict, ['final_f_val'])

    # Uncomment the following line to use the detailed plotting function
    plot_run_solvers(output_list)

    ################################################################################
    ################################################################################
    ################################################################################

    # For reference:
    # test_problems_list = ['rosenbrock',                      # 0
    #                     'powell',                            # 1
    #                     'well_conditioned_convex_quadratic', # 2
    #                     'ill_conditioned_convex_quadratic']  # 3

    # # SELECT PROBLEM
    # PROBLEM_NAME = test_problems_list[0]
    # INPUT_DIM = 12
    # PROBLEM_TUP = problems.test_problems.select_problem(problem_name=PROBLEM_NAME, input_dim=INPUT_DIM)
    # x0, obj = PROBLEM_TUP

    # # SOLVER CONFIG

    # REG_LAMBDA = 0.01
    # USE_HESS = True

    # INNER_USE_FULL_GRAD = True
    # REPROJECT_GRAD = False       # NOTE: makes no difference in the randomised proj variant

    # DIRECTION_STR = 'sd'      # NOTE: options: {'newton', 'sd'}

    # RANDOM_PROJ = True

    # ENSEMBLE = 'haar'
    # ALPHA = 0.01
    # T_INIT = 1
    # TAU = 0.5

    # MAX_ITER = np.inf
    # DERIV_BUDGET = 300

    # TOL = 1e-4
    # ITER_PRINT_GAP = 20

    # # Run solver(s)
    # solvers_list = []
    # output_list = []

    # SUBSPACE_NO_LIST = [(0, 0, 3),
    #                     (1, 1, 1)]

    # for SUBSPACE_NO_GRADS, SUBSPACE_NO_UPDATES, SUBSPACE_NO_RANDOM in SUBSPACE_NO_LIST:
    #     SUBSPACE_DIM = SUBSPACE_NO_GRADS + SUBSPACE_NO_UPDATES + SUBSPACE_NO_RANDOM
    #     RANDOM_PROJ_DIM = SUBSPACE_DIM # NOTE: this NEED NOT be the case...
    #     for _ in range(1): # number of runs per solver
    #         SOLVER_CONFIG = ProjectedCommonDirectionsConfig(obj=obj,
    #                                                         subspace_no_grads=SUBSPACE_NO_GRADS,
    #                                                         subspace_no_updates=SUBSPACE_NO_UPDATES,
    #                                                         subspace_no_random=SUBSPACE_NO_RANDOM,
    #                                                         inner_use_full_grad=INNER_USE_FULL_GRAD,
    #                                                         direction_str=DIRECTION_STR,
    #                                                         reg_lambda=REG_LAMBDA,
    #                                                         use_hess=USE_HESS,
    #                                                         random_proj=RANDOM_PROJ,
    #                                                         random_proj_dim=RANDOM_PROJ_DIM,
    #                                                         reproject_grad=REPROJECT_GRAD,
    #                                                         ensemble=ENSEMBLE,
    #                                                         alpha=ALPHA,
    #                                                         t_init=T_INIT,
    #                                                         tau=TAU,
    #                                                         tol=TOL,
    #                                                         max_iter=MAX_ITER,
    #                                                         deriv_budget=DERIV_BUDGET,
    #                                                         iter_print_gap=ITER_PRINT_GAP,
    #                                                         verbose=True)
            
    #         solvers_list.append(ProjectedCommonDirections(config=SOLVER_CONFIG))

    #         # optimiser = ProjectedCommonDirections(config=SOLVER_CONFIG)
    #         # output = optimiser.optimise(x0)
    #         # output_list.append(output)

    # AVG_RESULTS_ATTRS = ['final_f_val']
    # avg_results_dict = average_solver_runs(PROBLEM_TUP,
    #                                     solvers_list,
    #                                     no_runs=2,
    #                                     result_attrs=AVG_RESULTS_ATTRS)



    # PLOTTING
    # plotting.plotting.plot_solver_averages(avg_results_dict, ['final_f_val'])


    # plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list,
    #                                          deriv_evals_axis=True)

    # plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list,
    #                                          deriv_evals_axis=False)

    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['update_norms'],
    #                                            log_plot=True)

    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['direction_norms'],
    #                                            log_plot=True)

    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['full_grad_norms',
    #                                                        'proj_grad_norms'],
    #                                            log_plot=True)

    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['angles_to_full_grad'],
    #                                            log_plot=False)

    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['cond_nos'],
    #                                            log_plot=True)

    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['P_ranks'],
    #                                            log_plot=False)

    # plotting.plotting.plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                            attr_names=['P_norms'],
    #                                            log_plot=True)

    # plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                                 attr_names=['f_vals', 'P_ranks'],
    #                                                 log_plots=[True, False])

    # plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                                 attr_names=['proj_grad_norms', 'P_ranks'],
    #                                                 log_plots=[True, False])

    # plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                                 attr_names=['f_vals', 'P_norms'],
    #                                                 log_plots=[True, True])

    # plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                                 attr_names=['P_norms', 'P_ranks'],
    #                                                 log_plots=[True, False])

    # plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                                 attr_names=['update_norms', 'P_ranks'],
    #                                                 log_plots=[True, False])

    # plotting.plotting.twin_plot_scalar_vs_iteration(solver_outputs=output_list,
    #                                                 attr_names=['f_vals', 'angles_to_full_grad'],
    #                                                 log_plots=[True, False])

    plt.show()

if __name__ == '__main__':
    main()