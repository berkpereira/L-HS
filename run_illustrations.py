import os
import plotting.plotting
from running import running
import results.results_utils
from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import matplotlib.pyplot as plt

def main():
    running.soft_window_clear()
    # Choose problem
    test_problems_list = ['rosenbrock_single',                 # 0, n even
                        #   'rosenbrock_multiple',               # 1, n even
                        #   'powell',                            # 2, n multiple of 4
                        #   'well_conditioned_convex_quadratic', # 3, any n
                        #   'ill_conditioned_convex_quadratic',  # 4, any n
                        #   'POWELLSG',                          # 5,  n in {4, 8, 16, 20, 36, 40, 60, 80, 100, 500}
                        #   'POWER',                             # 6,  n in {10, 20, 30, 50, 75, 100, 500}
                        #   'NONDIA',                            # 7,  n in {10, 20, 30, 50, 90, 100, 500}
                        #   'NCB20B',                            # 8,  n in {21, 22, 50, 100, 180, 500}
                        #   'OSCIGRAD',                          # 9,  n in {2, 5, 10, 15, 25, 100}
                        #   'TRIDIA',                            # 10, n in {10, 20, 30, 50, 100, 500}
                        #   'OSCIPATH',                          # 11, n in {2, 5, 10, 25, 100, 500}
                        #   'YATP2LS',                           # 12, n in {2, 10, 50, 100, 200, 350}
                        #   'PENALTY2'                           # 13, n in {4, 10, 50, 100, 200, 500}
                          ]                          
    problem_name = test_problems_list[0]
    input_dim = 100
    extended_problem_name = problem_name + '_n' + str(input_dim)
    problem_tup = running.get_problem(problem_name, input_dim)
    NO_RUNS = 5

################################################################################
################################################################################
    order = 'sd'
    experiment_str = 'granular'
    
    if order == 'sd':
        if experiment_str == 'granular':
            solver_names = [ # NOTE: select solvers
                # '1d.0.0',
                # '1d.1d.0',
                # '2.0.0',
                # '2.2.0',
                # '3.0.0',
                # '3.3.0',
                # '4.0.0',
                # '4.4.0',

                # '1d.0.10',
                # '1d.1d.10',
                # '2.0.10',
                # '2.2.10',
                # '3.0.10',
                # '3.3.10',
                # '4.0.10',
                # '4.4.10',
                
                # '5.0.0',
                # '5.5.0',
                # '7.0.0',
                # '7.7.0',
                # '10.0.0',
                # '10.10.0',
                # '15.0.0',
                # '15.15.0',

                # '5.0.10',
                # '5.5.10',
                # '7.0.10',
                # '7.7.10',
                # '10.0.10',
                # '10.10.10',
                # '15.0.10',
                # '15.15.10',

                # Strings of the form '1d.0.{0, 1, 2, 5, 7, 10}'
                # '1d.0.0',
                # '1d.0.1',
                # '1d.0.2',
                # '1d.0.5',
                # '1d.0.7',
                # '1d.0.10',

                # Strings of the form '5.5.{0,2,5,10,20,40}'
                # '5.5.0',
                # '5.5.2',
                # '5.5.5',
                # '5.5.10',
                # '5.5.20',
                # '5.5.40',

                # Strings of the form '10.10.{0,5,10,20}'
                # '10.10.0',
                # '10.10.5',
                # '10.10.10',
                # '10.10.20',

                # Strings of the form '20.20.{0,5,10,20}'
                '20.20.0',
                '20.20.5',
                '20.20.10',
                '20.20.20',
                
                # Strings of the form '1d.{0,1d}.{0,5,10,20,40}'
                # '1d.0.0',
                # '1d.1d.0',
                # '1d.0.5',
                # '1d.1d.5',
                # '1d.0.10',
                # '1d.1d.10',
                # '1d.0.20',
                # '1d.1d.20',
                # '1d.0.40',
                # '1d.1d.40',
            ]
        passable_name = 'default_illustrations_sd'
    elif order == 'newton':
        if experiment_str in ['orth_Pk', 'haar_gauss']:
            solver_names = [
                'solver1',
                'solver2',
                'solver3',
                'solver4',
            ]
        elif experiment_str == 'sketch_size':
            solver_names = [
                'solver1',
                'solver2',
                'solver3',
                'solver4',
                'solver5',
            ]
        passable_name = 'default_illustrations_newton'
    elif order == 'quasi_newton':
        solver_names = [
            'solver1',
            'solver2',
            'solver3',
            'solver4',
        ]
        passable_name = 'default_illustrations_quasi_newton'
    
    CONFIG_PATH_LIST = [[order, experiment_str, name] for name in solver_names]
    

    RUN          = False
    SAVE_RESULTS = False

    PLOT         = True
    SAVE_FIG     = True
    INCLUDE_SOLVER_NAMES = True
    FOR_APPENDIX = False
    
    FIGSIZE = (5.9, 2.4) # NOTE: the one most used
    # FIGSIZE = (5.9, 2.0) # NOTE: if not much space is required
    # FIGSIZE = (5.9, 2.5) # NOTE: if a bit more (vertical) space is required
    LABEL_NCOL = 1



################################################################################
################################################################################

    configs_list = []
    for config_path in CONFIG_PATH_LIST:
        config = running.combine_configs(extended_problem_name,
                                         config_path=config_path,
                                         passable_name=passable_name)
        configs_list.append(config)
                    
    solvers_list = [ProjectedCommonDirections(config) for config in configs_list]

    # Run and store results
    results_attrs = ['final_f_val']
    if RUN:
        results_dict = running.run_solvers_single_prob(problem_tup, solvers_list,
                                                    no_runs=NO_RUNS,
                                                    result_attrs=results_attrs,
                                                    save_results=SAVE_RESULTS)

    if PLOT:
        outputs_list = results.results_utils.generate_illustrations(config_path_list=CONFIG_PATH_LIST,
                                                                    extended_problem_name=extended_problem_name,
                                                                    no_runs=NO_RUNS,
                                                                    passable_config_name=passable_name)
        
        fig = plotting.plotting.plot_loss_vs_iteration(solver_outputs=outputs_list,
                                                       include_Pk_orth=False,
                                                       include_sketch_size=False,
                                                       include_ensemble=False,
                                                       figsize=FIGSIZE,
                                                       label_ncol=LABEL_NCOL)
        plt.show()

    if SAVE_FIG:
        if not PLOT:
            raise Exception('Trying to save plot but none has been created!')
        file_path = results.results_utils.generate_pdf_file_name(CONFIG_PATH_LIST,
                                                                 plot_type='illustration',
                                                                 for_appendix=FOR_APPENDIX,
                                                                 include_solver_names=INCLUDE_SOLVER_NAMES,
                                                                 solver_name_list=solver_names)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fig.savefig(fname=file_path)

if __name__ == '__main__':
    main()