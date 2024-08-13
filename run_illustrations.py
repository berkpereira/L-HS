from running import running
import results.results_utils
from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import matplotlib.pyplot as plt

def main():
    running.soft_window_clear()
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


################################################################################
################################################################################

    CONFIG_PATH_LIST = [
        ['sd', 'orth_Pk', 'solver1'],
        ['sd', 'orth_Pk', 'solver2'],
        ['sd', 'orth_Pk', 'solver3'],
        ['sd', 'orth_Pk', 'solver4']
        ]
    
    NO_RUNS = 5

    NORMALISE_LOSS = True
    
    SAVE_RESULTS = False
    
    SAVE_FIG = False
    LABEL_NCOL = 1

    passable_name = 'default_illustrations'

################################################################################
################################################################################

    configs_list = []
    for config_path in CONFIG_PATH_LIST:
        configs_list.append(running.combine_configs(extended_problem_name, config_path=config_path, passable_name=passable_name))
                    
    solvers_list = [ProjectedCommonDirections(config) for config in configs_list]

    # Run and store results
    results_attrs = ['final_f_val']
    results_dict = running.run_solvers_single_prob(problem_tup, solvers_list,
                                                   no_runs=NO_RUNS,
                                                   result_attrs=results_attrs, save_results=SAVE_RESULTS)

    # Plot
    # plotting.plotting.plot_solver_averages(results_dict, ['final_f_val'])
    # plotting.plotting.plot_run_histograms(results_dict['raw_results'],
    #                                       attr_names=results_attrs)

    # (detailed plots, each individual run represented)
    fig = running.plot_run_solvers(results_dict, NORMALISE_LOSS,
                                   include_Pk_orth=True,
                                   include_sketch_size=True,
                                   figsize=(5.9, 2.4),
                                   label_ncol=LABEL_NCOL)

    plt.show()
    if SAVE_FIG:
        fig.savefig(fname=results.results_utils.generate_pdf_file_name(CONFIG_PATH_LIST, 'illustration'))

if __name__ == '__main__':
    main()