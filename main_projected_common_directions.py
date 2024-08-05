from running import running
from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import matplotlib.pyplot as plt

def main():
    running.soft_window_clear()
    running.set_seed(42)

    # Choose problem
    test_problems_list = ['rosenbrock_single',                 # 0, n even
                          'rosenbrock_multiple',               # 1, n even
                          'powell',                            # 2, n multiple of 4
                          'well_conditioned_convex_quadratic', # 3, any n
                          'ill_conditioned_convex_quadratic',  # 4, any n
                          'POWELLSG',                          # 5,  n in {4, 8, 16, 20, 36, 40, 60, 80, 100, 500}
                          'POWER',                             # 6,  n in {10, 20, 30, 50, 75, 100, 500}
                          'NONDIA',                            # 7,  n in {10, 20, 30, 50, 90, 100, 500}
                          'NCB20B',                            # 8,  n in {21, 22, 50, 100, 180, 500}
                          'OSCIGRAD',                          # 9,  n in {2, 5, 10, 15, 25, 100}
                          'TRIDIA',                            # 10, n in {10, 20, 30, 50, 100, 500}
                          'OSCIPATH',                          # 11, n in {2, 5, 10, 25, 100, 500}
                          'YATP2LS',                           # 12, n in {2, 10, 50, 100, 200, 350}
                          'PENALTY2']                          # 13, n in {4, 10, 50, 100, 200, 500}
    problem_name = test_problems_list[0]
    input_dim = 100
    extended_problem_name = problem_name + '_n' + str(input_dim)
    problem_tup = running.get_problem(problem_name, input_dim)
    NORMALISE_LOSS = True
    SAVE_RESULTS = False

    passable_name = 'passable2'
    configs_list = [running.combine_configs(extended_problem_name, 'solver0', passable_name),
                    running.combine_configs(extended_problem_name, 'solver1', passable_name),
                    running.combine_configs(extended_problem_name, 'solver2', passable_name),
                    running.combine_configs(extended_problem_name, 'solver3', passable_name),]
                    
    solvers_list = [ProjectedCommonDirections(config) for config in configs_list]

    # Run and store results
    results_attrs = ['final_f_val']
    results_dict = running.run_solvers_single_prob(problem_tup, solvers_list,
                                                   no_runs=3,
                                                   result_attrs=results_attrs, save_results=SAVE_RESULTS)

    # Plot
    # plotting.plotting.plot_solver_averages(results_dict, ['final_f_val'])
    # plotting.plotting.plot_run_histograms(results_dict['raw_results'],
    #                                       attr_names=results_attrs)

    # (detailed plots, each individual run represented)
    running.plot_run_solvers(results_dict, NORMALISE_LOSS,
                            suppress_dir=True,
                            suppress_sketch_size=True)

    plt.show()

if __name__ == '__main__':
    main()