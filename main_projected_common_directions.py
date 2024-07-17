import problems.test_problems
import autograd.numpy as np
from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
from solver_configs.projected_common_directions_configs import solver_variants_dict
from solver_configs.passable_configs import passable_variants_dict

from solvers.utils import average_solver_runs
import plotting
import matplotlib.pyplot as plt
import os

"""

TODO:
TODO:
TODO:
TODO:
Make a suite of configuration sets in well-organised folders/files, perhaps
organised by numerical study. E.G., one such file could be dedicated to solver
configurations used in a numerical study assessing the effect of changing
the random_proj_dim (the 'dimension' of S_k).

TODO:
TODO:
TODO:
ADD IN 1 OR 2 TEST PROBLEMS FROM CUTEst
See the timed out problems from the Cartis and Roberts 2023 paper on scalable DFO.


"""

def set_seed(seed):
    np.random.seed(42)

def soft_window_clear():
    for _ in range(40):
        print()

# This function combines the solver and passable configurations into a proper
# configuration object (ProjectedCommonDirectionsConfig) to be subsequently used.
def combine_configs(problem_name: str, input_dim: int, solver_name: str, passable_name: str):
    config_dict = {'obj': problems.test_problems.select_problem(problem_name, input_dim)[1],
                   **solver_variants_dict[solver_name],
                   **passable_variants_dict[passable_name]}
    return ProjectedCommonDirectionsConfig(**config_dict)

def get_problem(problem_name, input_dim):
    return problems.test_problems.select_problem(problem_name=problem_name, input_dim=input_dim)

def configure_solver(obj, subspace_no_grads, subspace_no_updates,
                     subspace_no_random, random_proj_dim, **kwargs):
    subspace_dim = subspace_no_grads + subspace_no_updates + subspace_no_random
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
    results_dict = average_solver_runs(problem_tup, solvers_list, no_runs, result_attrs)
    return results_dict

def plot_run_solvers(output_dict):
    # Extract all SolverOutput objects from a results dictionary.
    output_list = [solver_output for _, solver_outputs in output_dict['raw_results'] for solver_output in solver_outputs]
    
    # Uncomment the following lines to 'select' plotting functions
    plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list,
                                             deriv_evals_axis=True,
                                             normalise_deriv_evals_vs_dimension=True,
                                             normalise_P_k_dirs_vs_dimension=False,
                                             normalise_S_k_dirs_vs_dimension=False)
    plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list,
                                             deriv_evals_axis=False,
                                             normalise_P_k_dirs_vs_dimension=True,
                                             normalise_S_k_dirs_vs_dimension=True)
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

def main():
    soft_window_clear()
    set_seed(42)

    # Choose problem
    test_problems_list = ['rosenbrock_single',                 # 0
                          'rosenbrock_multiple',               # 1
                          'powell',                            # 2
                          'well_conditioned_convex_quadratic', # 3
                          'ill_conditioned_convex_quadratic',  # 4
                          'nondia']                            # 5
    problem_name = test_problems_list[5]
    input_dim = 100
    problem_tup = get_problem(problem_name, input_dim)
    x0, obj = problem_tup

    # subspace_no_list = [(1, 1, 2),
    #                     (2, 2, 3),
    #                     (3, 3, 2),
    #                     (3, 3, 1),
    #                     (4, 4, 2),
    #                     (4, 4, 0),
    #                     (0, 0, 5)]
    # solvers_list = []

    # for subspace_no_grads, subspace_no_updates, subspace_no_random in subspace_no_list:
    #     SUBSPACE_DIM_TOTAL = subspace_no_grads + subspace_no_updates + subspace_no_random
    #     TILDE_PROJ_DIM = 5
    #     solver = configure_solver(obj, subspace_no_grads, subspace_no_updates,
    #                             subspace_no_random, TILDE_PROJ_DIM, **fixed_solver_config_params)
    #     solvers_list.append(solver)

    passable_name = 'passable2' # NOTE: this should probably be the same for all solvers
    configs_list = [combine_configs(problem_name, input_dim, 'solver2', passable_name),
                    combine_configs(problem_name, input_dim, 'solver3', passable_name)]
    solvers_list = [ProjectedCommonDirections(config) for config in configs_list]

    # Run and store results
    results_attrs = ['final_f_val']
    results_dict = run_solvers(problem_tup, solvers_list, no_runs=2,
                               result_attrs=results_attrs)

    # Plot
    plotting.plotting.plot_solver_averages(results_dict, ['final_f_val'])
    plotting.plotting.plot_run_histograms(results_dict['raw_results'],
                                          attr_names=results_attrs)

    # (detailed plots, each individual run represented)
    plot_run_solvers(results_dict)

    plt.show()

if __name__ == '__main__':
    main()