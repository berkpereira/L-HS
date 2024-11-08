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

    # RUNNING TESTS
    RUN          = False
    SAVE_RESULTS = False
    
    # GENERATING DATA PROFILES
    GENERATE_PROFILE   = False
    READ_PROFILE       = True
    SAVE_PROFILE_DICTS = True # (Only relevant if GENERATE_PROFILE is True a profile)

    # PLOTTING DATA PROFILE
    PLOT_PROFILE         = True
    SAVE_FIG             = True
    INCLUDE_SOLVER_NAMES = True
    FOR_APPENDIX         = False
    EARLY_ITERATIONS     = False
    
    # PROFILE ATTRIBUTES
    ACCURACY = 1e-2
    LABEL_NCOL = 2
    PLOT_MAX_EQUIV_GRAD = 150      # NOTE: for SD
    # PLOT_MAX_EQUIV_GRAD = 2_800  # NOTE: for Newton
    # FIGSIZE = (5.9, 2.2)         # NOTE: the most commonly used (by me) size
    FIGSIZE = (5.9, 2.5)           # NOTE: for larger stuff (maybe benchmarks)

################################################################################
################################################################################
    order = 'sd'
    experiment_str = 'benchmarks'
    
    if order == 'sd':
        if experiment_str == 'benchmarks':
            solver_names = [
                'full-space-SD',
                # '1d.0.2.5',
                # '1d.0.2.20',
                # # '1d.0.2.10',
                # '5.5.10',
                # # '5.5.20',
                # '10.10.10',
                # # # '20.20.5',
                # '20.20.10',
                # # 'lee-1d.0.2',
                # # 'lee-5.5.10',
                # # 'lee-5.5.20',
                # # 'lee-10.10.10',
                # # 'lee-20.20.5',
                # # 'lee-20.20.10',
                '0.0.5',
                # '0.0.10',
                # '0.0.20',
                # # '0.0.30',
                # # '0.0.50',

                # Strings of the form '2.0.0.{2,3}'
                '2.0.0.2',
                '2.0.0.3',
                
                # Strings of the form '2.2.0.{2,3}'
                '2.2.0.2',
                '2.2.0.3',
                
                # Strings of the form '2.0.1.{2,3}'
                '2.0.1.2',
                '2.0.1.3',

                # Strings of the form '2.2.1.{2,3}'
                '2.2.1.2',
                '2.2.1.3',

                # Strings of the form '2.0.2.{2,3}'
                '2.0.2.2',
                '2.0.2.3',

                # Strings of the form '2.2.2.{2,3}'
                '2.2.2.2',
                '2.2.2.3',
            ]
        PASSABLE_NAME = 'default_data_profiles_sd'
    elif order == 'newton':
        if experiment_str == 'benchmarks':
            solver_names = [
            'full_space_Newton',
            '0.0.5',
            '0.0.10',
            '1d.0.2.20',
            '1d.0.2.40',
            '5.5.10.20',
            '5.5.10.40',
            '10.10.10.20',
            '10.10.10.40',
            '0.0.20',
            '0.0.50',
            'lee1d.0.2',
            'lee5.5.10',
            'lee10.10.10',
            ]
        elif experiment_str == 'sketch_size':
            solver_names = [
                'solver1',
                'solver2',
                'solver3',
                'solver4',
                'solver5',
            ]            
        elif experiment_str == 'haar_gauss':
            solver_names = [
                'solver1',
                'solver2',
                'solver3',
                'solver4',
            ]
        PASSABLE_NAME = 'default_data_profiles_newton'
    else:
        raise ValueError(f'Unrecognised order string {order}.')

    CONFIG_PATH_LIST = [[order, experiment_str, name] for name in solver_names]

################################################################################
################################################################################
    
    if GENERATE_PROFILE and READ_PROFILE:
        raise Exception('Cannot have both GENERATE_PROFILE and READ_PROFILE at the same time!')

    # Read in the 20 problems selected for small data profiles.
    if PROFILE == 'SMALL':
        with open('problems/small_profile_problems.json', 'r') as f:
            problem_name_list = json.load(f)
    elif PROFILE == 'LARGE':
        with open('problems/large_profile_problems.json', 'r') as f:
            problem_name_list = json.load(f)
    else:
        raise Exception("Profile problem set not recognised! (must be either 'SMALL' or 'LARGE')")

    if RUN and GENERATE_PROFILE:
        raise Exception('Cannot have both RUN and GENERATE_PROFILE at the same time!')

    if RUN:
        running.running.run_solvers_multiple_prob(extended_problem_name_list=problem_name_list,
                                                    config_path_list=CONFIG_PATH_LIST,
                                                    passable_name=PASSABLE_NAME,
                                                    no_runs=NO_RUNS,
                                                    save_results=SAVE_RESULTS)
        os.system('say running done!')

    ################################################################################
    ################################################################################
    configs_list = []
    for config_path in CONFIG_PATH_LIST:
        for problem_name in problem_name_list:
            config = running.running.combine_configs(extended_problem_name=problem_name,
                                                    config_path=config_path,
                                                    passable_name=PASSABLE_NAME,
                                                    ignore_problem=True)
            configs_list.append(config)

    print('Configs done')
    
    if GENERATE_PROFILE:
        success_dict = results.results_utils.generate_data_profiles(problem_name_list,
                                                                    configs_list, accuracy=ACCURACY,
                                                                    max_equiv_grad=PLOT_MAX_EQUIV_GRAD,
                                                                    save_profiles=SAVE_PROFILE_DICTS,
                                                                    large=(PROFILE=='LARGE'))
        print('Data profiles done')
    elif READ_PROFILE:
        success_dict = results.results_utils.load_data_profiles(config_list=configs_list,
                                                                results_dir='results',
                                                                large=(PROFILE=='LARGE'))


    if PLOT_PROFILE:
        fig = plotting.plotting.plot_data_profiles(success_dict,
                                                   include_Pk_orth=False,
                                                   include_sketch_size=True,
                                                   include_ensemble=False,
                                                   figsize=FIGSIZE,
                                                   label_ncol=LABEL_NCOL,
                                                   log_axis=False)
        os.system('say plot generation finished!')
        plt.show()
        if SAVE_FIG:
            plot_type = f'{PROFILE}_profile'
            file_path = results.results_utils.generate_pdf_file_name(CONFIG_PATH_LIST,
                                                                     plot_type=plot_type,
                                                                     accuracy=ACCURACY,
                                                                     for_appendix=FOR_APPENDIX,
                                                                     include_solver_names=INCLUDE_SOLVER_NAMES,
                                                                     solver_name_list=solver_names,
                                                                     early_iterations=EARLY_ITERATIONS)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            fig.savefig(fname=file_path)
            print()
        

if __name__ == '__main__':
    main()