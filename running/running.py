import problems.test_problems
import autograd.numpy as np
from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
from solver_configs.projected_common_directions_configs import solver_variants_dict
from solver_configs.passable_configs import passable_variants_dict
import results.results_utils
from problems.test_problems import select_problem

from solvers.utils import average_solver_runs, update_best_known_result, problem_name_dim_tuple_from_json_name
import plotting
import matplotlib.pyplot as plt
import os

def set_seed(seed):
    np.random.seed(42)

def soft_window_clear():
    os.system('clear')

# This function combines the solver and passable configurations into a proper
# configuration object (ProjectedCommonDirectionsConfig) to be subsequently used.
def combine_configs(extended_problem_name: str, solver_name: str,
                    passable_name: str):
    problem_name, input_dim = problem_name_dim_tuple_from_json_name(extended_problem_name)
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

def run_solvers_single_prob(problem_tup, solvers_list, no_runs,
                            result_attrs=['final_f_val'], save_results=False):
    
    results_dict = average_solver_runs(problem_tup, solvers_list, no_runs, result_attrs)
    
    # Update best-known results if a new best value is found
    for solver, solver_outputs in results_dict['raw_results']:
        for output in solver_outputs:
            if solver.obj.f_sol is None or output.final_f_val < solver.obj.f_sol:
                update_best_known_result(solver.obj.name, output.final_f_val)
            
            # Save results if required
            if save_results:
                results.results_utils.save_solver_output(solver.obj.name, str(solver.config), output)
    
    return results_dict

# This function is simpler than averaging ones, but more flexible in allowing
# for multiple solvers along with multiple problems.
def run_solvers_multiple_prob(extended_problem_name_list, solver_name_list,
                              passable_name, no_runs, 
                              save_results=False):
    for problem_name in extended_problem_name_list:
        problem_tuple = select_problem(problem_name, extended_name=True)
        x0, obj = problem_tuple
        for solver_name in solver_name_list:
            full_config = combine_configs(problem_name, solver_name,
                                          passable_name)
            solver = ProjectedCommonDirections(config=full_config)
            for _ in range(no_runs):
                output = solver.optimise(x0)
                if solver.obj.f_sol is None or output.final_f_val < solver.obj.f_sol:
                    update_best_known_result(solver.obj.name, output.final_f_val)
                
                # Save results if requested
                if save_results:
                    results.results_utils.save_solver_output(solver.obj.name,
                                                             str(solver.config),
                                                             output)




def plot_run_solvers(output_dict, normalise_loss,
                     suppress_c_const=False,
                     suppress_N_try=False,
                     suppress_dir=False,
                     suppress_sketch_size=False):
    # Extract all SolverOutput objects from a results dictionary.
    output_list = [solver_output for _, solver_outputs in output_dict['raw_results'] for solver_output in solver_outputs]
    
    # Uncomment the following lines to 'select' plotting functions
    plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list,
                                             deriv_evals_axis=True,
                                             normalise_deriv_evals_vs_dimension=True,
                                             normalise_P_k_dirs_vs_dimension=True,
                                             normalise_S_k_dirs_vs_dimension=True,
                                             normalise_loss_data=normalise_loss,
                                             suppress_c_const=suppress_c_const,
                                             suppress_N_try=suppress_N_try,
                                             suppress_dir=suppress_dir,
                                             suppress_sketch_size=suppress_sketch_size)
    plotting.plotting.plot_loss_vs_iteration(solver_outputs=output_list,
                                             deriv_evals_axis=False,
                                             normalise_P_k_dirs_vs_dimension=True,
                                             normalise_S_k_dirs_vs_dimension=True,
                                             normalise_loss_data=normalise_loss,
                                             suppress_c_const=suppress_c_const,
                                             suppress_N_try=suppress_N_try,
                                             suppress_dir=suppress_dir,
                                             suppress_sketch_size=suppress_sketch_size)
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

def plot_data_profiles(problem_name_list: list, solver_config_list: list,
                       accuracy: float, max_equiv_grad: int):
    success_dict = results.results_utils.generate_data_profiles(problem_name_list,
                                                                solver_config_list,
                                                                accuracy,
                                                                max_equiv_grad)
    plotting.plotting.plot_data_profiles(success_dict)
    

def run_average_solvers(problem_tup, solvers_list, no_runs, result_attrs):
    avg_results_dict = average_solver_runs(problem_tup, solvers_list, no_runs, result_attrs)
    return avg_results_dict
