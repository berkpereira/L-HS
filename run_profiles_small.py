from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import running.running
import results.results_utils
import plotting.plotting
import matplotlib.pyplot as plt
import json
import os

def main():
    # RUNNING
    RUN = False
    SAVE_RESULTS = False
    NO_RUNS = 10
    
    # PLOTTING
    PLOT_PROFILE = True
    SAVE_FIG = True
    FOR_APPENDIX = False

    ACCURACY = 1e-2
    PLOT_MAX_EQUIV_GRAD = 50
    LABEL_NCOL = 2
    FIGSIZE = (5.9, 2.2)

################################################################################
################################################################################
    order = 'sd'
    experiment_str = 'use_momentum'
    solver_numbers = [
        1,
        2,
        3,
        4,
    ]
    CONFIG_PATH_LIST = [[order, experiment_str, f'solver{i}'] for i in solver_numbers]

    PASSABLE_NAME = 'default_data_profiles'

################################################################################
################################################################################
    # Read in the 20 problems selected for small data profiles.
    with open('problems/small_profile_problems.json', 'r') as f:
        problem_name_list = json.load(f)

    if RUN and PLOT_PROFILE:
        raise Exception('Cannot have both RUN and PLOT_PROFILE at the same time!')

    if RUN:
        running.running.run_solvers_multiple_prob(extended_problem_name_list=problem_name_list,
                                                    config_path_list=CONFIG_PATH_LIST,
                                                    passable_name=PASSABLE_NAME,
                                                    no_runs=NO_RUNS,
                                                    save_results=SAVE_RESULTS)
        os.system('say running done!')

    ################################################################################
    ################################################################################

    if PLOT_PROFILE:
        configs_list = []
        for config_path in CONFIG_PATH_LIST:
            for problem_name in problem_name_list:
                config = running.running.combine_configs(extended_problem_name=problem_name,
                                                        config_path=config_path,
                                                        passable_name=PASSABLE_NAME,
                                                        ignore_problem=True)
                configs_list.append(config)

        print('Configs done')
        success_dict = results.results_utils.generate_data_profiles(problem_name_list,
                                                                    configs_list, accuracy=ACCURACY,
                                                                    max_equiv_grad=PLOT_MAX_EQUIV_GRAD)
        fig = plotting.plotting.plot_data_profiles(success_dict,
                                                   include_Pk_orth=False,
                                                   include_sketch_size=True,
                                                   include_ensemble=False,
                                                   figsize=FIGSIZE,
                                                   label_ncol=LABEL_NCOL)
        plt.show()
        if SAVE_FIG:
            file_path = results.results_utils.generate_pdf_file_name(CONFIG_PATH_LIST, plot_type='small_profile', accuracy=ACCURACY, for_appendix=FOR_APPENDIX)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            fig.savefig(fname=file_path)
            print()
        

if __name__ == '__main__':
    main()