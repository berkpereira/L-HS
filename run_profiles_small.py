from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import running.running
import results.results_utils
import plotting.plotting
import matplotlib.pyplot as plt
import json
import os

def main():
    os.system('clear')
    NO_RUNS = 10


    # NOTE: WHICH PROFILE problem set?
    PROFILE = 'LARGE' # \in {'SMALL', 'LARGE'}

    # RUNNING
    RUN          = True
    SAVE_RESULTS = True
    
    # PLOTTING
    PLOT_PROFILE = False
    SAVE_FIG     = False
    FOR_APPENDIX = False
    INCLUDE_SOLVER_NAMES_IN_FIG_FILE_PATH = True

    ACCURACY = 1e-2
    # PLOT_MAX_EQUIV_GRAD = 150 # NOTE: for SD
    # PLOT_MAX_EQUIV_GRAD = 2_800 # NOTE: for Newton
    LABEL_NCOL = 2
    FIGSIZE = (5.9, 2.2)

################################################################################
################################################################################
    # order = 'sd'
    # experiment_str = 'benchmarks'
    # # # NOTE: THE BELOW IS FOR SD GRANULAR/BENCHMARKS
    # solver_names = [
    #     'full-space-SD',
    #     # '1d.0.2',
    #     '1d.0.2.5',
    #     '1d.0.2.10',
    #     '5.5.10',
    #     # # '5.5.20',
    #     '10.10.10',
    #     # # '20.20.5',
    #     '20.20.10',
    #     # # 'lee-1d.0.2',
    #     # # 'lee-5.5.10',
    #     # # 'lee-5.5.20',
    #     # # 'lee-10.10.10',
    #     # # 'lee-20.20.5',
    #     # # 'lee-20.20.10',
    #     '0.0.5',
    #     # # '0.0.10',
    #     '0.0.20',
    #     # # '0.0.30',
    #     '0.0.50',
    # ]
    
    order = 'newton'
    experiment_str = 'benchmarks'
    solver_names = [
    'full_space_Newton',
    '5.5.10.20',
    '5.5.10.40',
    '10.10.10.20',
    '10.10.10.40',
    '0.0.5',
    '0.0.10',
    '0.0.20',
    '0.0.50',
    '1d.0.2.20',
    '1d.0.2.40',
    ]
    
    CONFIG_PATH_LIST = [[order, experiment_str, name] for name in solver_names]

    PASSABLE_NAME = 'default_data_profiles_newton'

################################################################################
################################################################################
    # Read in the 20 problems selected for small data profiles.
    if PROFILE == 'SMALL':
        with open('problems/small_profile_problems.json', 'r') as f:
            problem_name_list = json.load(f)
    elif PROFILE == 'LARGE':
        with open('problems/large_profile_problems.json', 'r') as f:
            problem_name_list = json.load(f)
    else:
        raise Exception("Profile problem set not recognised! (must be either 'SMALL' or 'LARGE')")

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
                                                   include_sketch_size=False,
                                                   include_ensemble=False,
                                                   figsize=FIGSIZE,
                                                   label_ncol=LABEL_NCOL,
                                                   log_axis=False)
        plt.show()
        if SAVE_FIG:
            plot_type = f'{PROFILE}_profile'
            file_path = results.results_utils.generate_pdf_file_name(CONFIG_PATH_LIST,
                                                                     plot_type=plot_type,
                                                                     accuracy=ACCURACY,
                                                                     for_appendix=FOR_APPENDIX,
                                                                     include_solver_names=INCLUDE_SOLVER_NAMES_IN_FIG_FILE_PATH,
                                                                     solver_name_list=solver_names)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            fig.savefig(fname=file_path)
            print()
        

if __name__ == '__main__':
    main()