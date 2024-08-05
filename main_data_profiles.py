from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import running.running
import results.results_utils
import plotting.plotting

def main():

    RUN = False
    SAVE_RESULTS = False

    PROFILE = True
    PLOT = True

    SOLVER_NAME_LIST = ['solver3']
    PASSABLE_NAME = 'passable2'

    problem_name_list = ['rosenbrock_single_n100',
                         'POWER_n100',
                         'NONDIA_n100']

    NO_RUNS = 10

################################################################################
################################################################################

    if RUN and PROFILE:
        raise Exception('Cannot have both RUN and PROFILE at the same time!')

    configs_list = []
    for problem_name in problem_name_list:
        config = running.running.combine_configs(extended_problem_name=problem_name,
                                                 solver_name='solver3',
                                                 passable_name='passable2')
        configs_list.append(config)

    if RUN:
        running.running.run_solvers_multiple_prob(extended_problem_name_list=problem_name_list,
                                                solver_name_list=SOLVER_NAME_LIST,
                                                passable_name=PASSABLE_NAME,
                                                no_runs=NO_RUNS,
                                                save_results=SAVE_RESULTS)
    
################################################################################
################################################################################

    if PROFILE:
        success_dict = results.results_utils.generate_data_profiles(problem_name_list,
                                                                    configs_list, accuracy=0.5,
                                                                    max_equiv_grad=200)
        if PLOT:
            plotting.plotting.plot_data_profiles(success_dict)

if __name__ == '__main__':
    main()