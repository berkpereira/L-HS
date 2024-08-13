from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import running.running
import results.results_utils
import plotting.plotting
import matplotlib.pyplot as plt
import json
import os

def main():

    RUN = False
    SAVE_RESULTS = False
    NO_RUNS = 10
    
    PLOT_PROFILE = True
    SAVE_FIG = False

################################################################################
################################################################################

    CONFIG_PATH_LIST = [
        ['sd', 'orth_Pk', 'solver1'],
        ['sd', 'orth_Pk', 'solver2'],
        ['sd', 'orth_Pk', 'solver3'],
        ['sd', 'orth_Pk', 'solver4']
        ]

    PASSABLE_NAME = 'default'

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
                                                                    configs_list, accuracy=1e-2,
                                                                    max_equiv_grad=150)
        fig = plotting.plotting.plot_data_profiles(success_dict,
                                                   include_Pk_orth=True,
                                                   include_sketch_size=True,
                                                   figsize=(5.9, 2.6))
        plt.show()
        if SAVE_FIG:
            fig.savefig(fname=results.results_utils.generate_pdf_file_name(CONFIG_PATH_LIST, 'small_profile'))
        

if __name__ == '__main__':
    main()