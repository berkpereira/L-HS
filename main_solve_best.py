from running import running
from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import json
import re
import os

def read_json_problems(file_name: str):
    with open(file_name, 'r') as file:
        items_list = json.load(file)
    return items_list

def problem_tuple_from_json_name(json_prob_name: str):
    match = re.match(r"([A-Za-z0-9\-]+)_n(\d+)", json_prob_name)
    if match:
        return match.group(1), int(match.group(2))
    else:
        raise ValueError("String format is not as expected")

def main():
    running.soft_window_clear()
    running.set_seed(42)

    PROBLEM_NAMES_FILE = 'problems/matching_problems.json' # THIS HAS 388 PROBLEMS IN IT
    problem_names_list = read_json_problems(PROBLEM_NAMES_FILE)

    last_prob_sol_stored = 'SBRYBND_n100'
    
    last_index = problem_names_list.index(last_prob_sol_stored)
    print(f'Stopped at problem {last_prob_sol_stored}, with index {last_index}')
    print()
    _ = os.system(f'say last solved problem has index {last_index}')

    # Refer to OBSIDIAN WEEK 32 log notes on this!
    for json_prob_name in problem_names_list[92:]:
        # raise Exception('Check where we stopped by looking at the best known results file')
        problem_name, input_dim = problem_tuple_from_json_name(json_prob_name)
        
        if input_dim > 200:
            continue

        try:
            problem_tup = running.get_problem(problem_name, input_dim)
        except: # stuff can go wrong, weird stuff importing CUTEst problems...
            _ = os.system(f'say skipped problem {problem_name}... with dimension {input_dim}... an exception was encountered')
            continue
        
        _ = os.system(f'say about to attempt solving {problem_name}... with dimension {input_dim}')

        SAVE_RESULTS = False

        passable_name = 'solve_best'
        best_config = running.combine_configs(problem_name, input_dim, 'solve_best', passable_name)
        best_solver_list = [ProjectedCommonDirections(best_config)]

        # Run and store results
        results_attrs = ['final_f_val']

        try:
            results_dict = running.run_solvers(problem_tup, best_solver_list, no_runs=1,
                                                result_attrs=results_attrs, save_results=SAVE_RESULTS)
        except: # e.g. if a matrix is singular or something
            _ = os.system(f'say something went wrong in solving problem {problem_name} with dimension {input_dim}... skipping')
            continue

if __name__ == '__main__':
    main()