"""
This small script is intended to help randomly select a subset of the
suitable CUTEst problems for use throughout the numerical experiments of my
MMSC thesis.
The idea is, at each step of variant-tweaking, to have both a detailed
test on just one or two test functions, along with a small benchmarking exercise
of the variants in a small suite of CUTEst problems, which I aim to select here.

The expectation here is to use something like 15--20 problems.
"""
import os
import pycutest
import json
import solvers.utils
import random

random.seed(4)

def load_json_to_dict(file_path):
    with open(file_path, 'r') as file:
        data_dict = json.load(file)
    return data_dict

def write_json_file(json_file_name: str, string_list: list):
    with open(json_file_name, 'w') as json_file:
        json.dump(string_list, json_file)

def generate_chosen_dict():
    results_dict = load_json_to_dict('results/best_known_results.json')
    no_suitable = 0
    suitable_name_dim_dict = {}
    for extended_name in results_dict:
        name, input_dim = solvers.utils.problem_name_dim_tuple_from_json_name(extended_name)
        best_sol = results_dict[extended_name]
        if 100 <= input_dim <= 200 and best_sol < 1e4:
            # print(f'Name: {name}, dim: {input_dim}')
            # print(f'Best sol.: {results_dict[extended_name]}')
            if name in suitable_name_dim_dict:
                suitable_name_dim_dict[name].append(input_dim)
            else:
                suitable_name_dim_dict.update({name: [input_dim]})
            no_suitable += 1
    return suitable_name_dim_dict, no_suitable

def select_random_combinations(input_dict, num_combinations):
    # Select 5 keys uniformly at random from the dictionary
    selected_keys = random.sample(list(input_dict.keys()), num_combinations)
    
    # For each selected key, select one of the elements in the corresponding list value uniformly at random
    combinations = []
    for key in selected_keys:
        value = random.choice(input_dict[key])
        combinations.append(f"{key}_n{value}")
    
    return combinations

def main():
    os.system('clear')
    suitable_name_dim_dict, no_suitable = generate_chosen_dict()
    print(f'NO. SUITABLE PROBLEMS AS FILTERED: {no_suitable}')
    # print(suitable_name_dim_dict)

    WRITE_TO_FILE = False

    chosen_problems = select_random_combinations(suitable_name_dim_dict, 20)
    print(chosen_problems)

    if WRITE_TO_FILE:
        JSON_FILE_NAME = 'problems/small_profile_problems.json'
        write_json_file(JSON_FILE_NAME, chosen_problems)
    

if __name__ == '__main__':
    main()