import json
import os
import csv
import re
from ast import literal_eval
import hashlib
from datetime import datetime
from solvers.utils import SolverOutput, problem_name_dim_tuple_from_json_name
from solvers.projected_common_directions import ProjectedCommonDirections, ProjectedCommonDirectionsConfig
import problems.test_problems

MAPPING_FILE = 'results/config_mapping.json'

def get_run_id():
    # Generate a unique ID based on time when running the algorithm
    return hashlib.sha256(datetime.now().isoformat().encode()).hexdigest()[:10]

def get_hashed_filename(solver_config_str: str):
    # Generate a hash of the solver configuration string
    return hashlib.sha256(solver_config_str.encode()).hexdigest()[:10]

def save_mapping(solver_config_str, hashed_filename):
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r') as file:
            mapping = json.load(file)
    else:
        mapping = {}

    mapping[solver_config_str] = hashed_filename

    with open(MAPPING_FILE, 'w') as file:
        json.dump(mapping, file, indent=4)

# NOTE: THINK THIS FUNCTION BELOW IS NO LONGER NEEDED
def config_to_mapping_hash(solver_config: ProjectedCommonDirectionsConfig):
    # Load the mapping file
    solver_config_str = str(solver_config)
    if not os.path.exists(MAPPING_FILE):
        raise Exception('Mapping file does not exist.')

    with open(MAPPING_FILE, 'r') as file:
        mapping = json.load(file)

    if solver_config_str not in mapping:
        raise Exception('Solver configuration not found in the mapping file.')

    # Get the hashed file name
    hashed_filename = mapping[solver_config_str]
    return hashed_filename

def hash_to_config_str(hash: str) -> str:
    # Load the mapping file
    if not os.path.exists(MAPPING_FILE):
        raise Exception('Mapping file does not exist.')

    with open(MAPPING_FILE, 'r') as file:
        mapping = json.load(file)

    # Find the config string corresponding to the hash
    for config_str, mapped_hash in mapping.items():
        if mapped_hash == hash:
            return config_str

    raise Exception('Hash not found in the mapping file.')

def create_config_from_config_string(config_str: str, objective_instance) -> ProjectedCommonDirections:
    """
    Create an instance of ProjectedCommonDirectionsConfig from a string
    representation of it.

    Args:
        config_str (str): String representation of ProjectedCommonDirectionsConfig.
        objective_instance (Objective): An instance of the Objective class to be used in the configuration.

    Returns:
        ProjectedCommonDirectionsConfig: Initialised config instance.
    """
    # Remove the class name and parentheses
    config_str = config_str[len("ProjectedCommonDirectionsConfig("):-1]
    
    # Split the string into key-value pairs
    config_dict = {}
    key_value_pairs = config_str.split(', ')
    
    for pair in key_value_pairs:
        key, value = pair.split('=')
        key = key.strip()
        value = value.strip()
        
        # Handle specific cases for boolean and None
        if value in {"True", "False"}:
            value = value == "True"
        elif value == "None":
            value = None
        else:
            try:
                # Try to interpret the value as a literal (int, float, etc.)
                value = literal_eval(value)
            except (ValueError, SyntaxError):
                # If literal_eval fails, leave the value as a string
                pass

        config_dict[key] = value

    # Add the objective instance to the configuration dictionary
    config_dict['obj'] = objective_instance

    # Create the ProjectedCommonDirectionsConfig instance
    config = ProjectedCommonDirectionsConfig(**config_dict)
    
    return config


# NOTE: this functions keeps only KEEP_NO solver runs' data.
# These are prioritised vs any incoming data by the equivalent gradient
# evaluations budget of the run. The larger the budget, the higher the priority.
# NOTE: KEEP_NO is hard-coded here! For instance, Cartis' and Roberts' 2023
# paper uses KEEP_NO equal to 10 (see its page 39, final line).
def save_solver_output(problem_name: str, solver_config_str: str,
                       solver_output: SolverOutput, output_dir='results'):
    
    KEEP_NO = 10 # 10 is used in Cartis and Roberts 2023 scalable DFO paper
    
    # Create the directory for the problem if it does not exist
    problem_dir = os.path.join(output_dir, problem_name)
    os.makedirs(problem_dir, exist_ok=True)
    
    # Create a hashed file name for the CSV file
    hashed_filename = get_hashed_filename(solver_config_str)
    save_mapping(solver_config_str, hashed_filename)
    file_path = os.path.join(problem_dir, f'{hashed_filename}.csv')
    
    # Collect data to save
    data = {
        'run_id': get_run_id(),
        'deriv_evals': solver_output.deriv_evals,
        'equiv_grad_evals': (solver_output.deriv_evals / solver_output.solver.obj.input_dim),
        'equiv_grad_budget': solver_output.solver.config.equiv_grad_budget,
        'f_vals': solver_output.f_vals,
        'update_norms': solver_output.update_norms,
        'full_grad_norms': solver_output.full_grad_norms,
        'proj_grad_norms': solver_output.proj_grad_norms,
        'config_str': str(solver_output.solver.config)
    }
    
    # Load existing data if the file exists
    existing_data = []
    if os.path.isfile(file_path):
        with open(file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_data.append(row)

    # Group data by run_id
    grouped_data = {}
    for row in existing_data:
        run_id = row['run_id']
        if run_id not in grouped_data:
            grouped_data[run_id] = []
        grouped_data[run_id].append(row)
    
    # Add new run data to grouped data
    run_data = []
    for iter in range(len(solver_output.f_vals)):
        run_data.append({
            'run_id': data['run_id'],
            'equiv_grad_budget': data['equiv_grad_budget'],
            'iter': iter,
            'deriv_evals': data['deriv_evals'][iter] if data['deriv_evals'].size > iter else None,
            'equiv_grad_evals':  data['equiv_grad_evals'][iter] if data['equiv_grad_evals'].size > iter else None,
            'f_vals': data['f_vals'][iter],
            'update_norms': data['update_norms'][iter] if data['update_norms'].size > iter else None,
            'full_grad_norms': data['full_grad_norms'][iter] if data['full_grad_norms'].size > iter else None,                
            'proj_grad_norms': data['proj_grad_norms'][iter] if data['proj_grad_norms'].size > iter else None,
            'config_str': data['config_str'] if iter == 0 else None  # Only store config_str in the first row
        })
    grouped_data[data['run_id']] = run_data
    
    # Sort runs by equiv_grad_budget and keep only the top KEEP_NO
    sorted_runs = sorted(grouped_data.values(), key=lambda x: float(x[0]['equiv_grad_budget']), reverse=True)[:KEEP_NO]

    # Write the selected runs back to the file
    with open(file_path, mode='w', newline='') as file:
        fieldnames = ['run_id', 'equiv_grad_budget', 'iter', 'deriv_evals', 'equiv_grad_evals', 'f_vals', 'update_norms', 'full_grad_norms', 'proj_grad_norms', 'config_str']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in sorted_runs:
            for row in run:
                if row['config_str'] is None:
                    del row['config_str']  # Remove the key to avoid writing None
                writer.writerow(row)

def load_solver_results(problem_name: str, solver_config, output_dir='results'):
    """
    problem_name: str formatted such as 'ROSEN-BR_n2', where the part after the
    underscore specifies the ambient dimension of the problem. 
    """
    # Get the hashed file name
    # hashed_filename = config_to_mapping_hash(solver_config)
    hashed_filename = get_hashed_filename(str(solver_config))
    file_path = os.path.join(output_dir, problem_name, f'{hashed_filename}.csv')

    if not os.path.exists(file_path):
        raise Exception('Result file does not exist.')

    # Load the results from the CSV file
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        results = [row for row in reader]

    return results

def get_best_known_sol(extended_problem_name: str):
    """
    extended_problem_name: str, should be of format such as 'ROSENBR_n2'
    """
    # First, try one of the manually implemented problems (NOT CUTEst)
    name, input_dim = problem_name_dim_tuple_from_json_name(extended_problem_name)

    if name in problems.test_problems.manual_list:
        x0, obj = problems.test_problems.select_problem(name, input_dim)
        f_sol = obj.f_sol
        return f_sol

    # Otherwise, this is a CUTEst problem, so the following should work if
    # it has been attempted before.

    with open('results/best_known_results.json', 'r') as f:
        best_known_results = json.load(f)
    try:
        f_sol = best_known_results.get(extended_problem_name, None)
        if f_sol is None:
            raise Exception('Retrieving best known solution returned None !')
        return f_sol
    except:
        raise Exception('Could not retrieve best known problem objective value!')


def generate_data_profiles(problem_name_list: list, solver_config_list: list,
                           accuracy: float, max_equiv_grad: int):
    """
    problem_name_list: list. Each element should be a string formatted such
    as 'ROSEN-BR_n2', where the part after the underscore specifies the
    ambient dimension of the problem.

    solver_config_list: list where each element is an instance
    of ProjectedCommonDirectionsConfig.

    accuracy: float between 0 and 1

    max_equiv_grad: int, maximum equivalent gradient evaluation number to
    check for when reading results, as well as limit to the right end of
    the x axis when actually plotting the data profile 
    """
    if not (0 < accuracy < 1):
        raise ValueError('Input accuracy must be in the open interval (0, 1).')

    equiv_grad_list = [i for i in range(max_equiv_grad + 1)]
    
    # Need a counter for each solver.
    # As we read each run in a given solver, we increment the corresponding
    # counter in this list, up from 0. Represents the CARDINALITY of
    # set P in the usual definitions of data profiles.
    seen_counter_dict = {}

    # Success dictionary. Each solver has a list which maps one-to-one with
    # equiv_grad_list. All starts at zero. As we read runs by a solver, 
    # whenever some equiv_grad allowance yielded the required accuracy,
    # we increment by one all the elements in the corresponding list from that
    # one up to the end one.
    success_dict = {}

    for solver_config in solver_config_list:
        solver_config_str = str(solver_config)
        hash = get_hashed_filename(solver_config_str)
        seen_counter_dict[hash] = 0
        success_dict[hash] = [0] * len(equiv_grad_list)
        for problem_name in problem_name_list:
            f_sol = get_best_known_sol(problem_name)
            results = load_solver_results(problem_name, solver_config)
            current_run_id = None
            current_run_data = []
            for row in results:
                run_id = row['run_id']
                if run_id != current_run_id:
                    if current_run_data:
                        process_run_data(current_run_data, equiv_grad_list,
                                         success_dict[hash], f_sol, accuracy)
                        seen_counter_dict[hash] += 1
                        current_run_data = []
                    current_run_id = run_id
                current_run_data.append(row)
            if current_run_data:
                process_run_data(current_run_data, equiv_grad_list, success_dict[hash], f_sol, accuracy)
                seen_counter_dict[hash] += 1
    
    # Check that all counter values are the same
    counter_values = list(seen_counter_dict.values())
    if counter_values.count(counter_values[0]) != len(counter_values):
        raise Exception('Inconsistent run counts across solvers.')
    
    # Normalize success_dict values by the seen_counter_dict values
    for hash in success_dict:
        count = seen_counter_dict[hash]
        if count > 0:
            success_dict[hash] = [value / count for value in success_dict[hash]]
    
    # Add on informative accuracy key-value pair, for later use in plotting.
    success_dict.update({'accuracy': accuracy, 'equiv_grad_list': equiv_grad_list})
    return success_dict

# NOTE: THIS FUNCTION FOR CHECKING WHETHER ACCURACY HAS BEEN MET,
# SEE WHERE IT IS CALLED ABOVE IN generate_data_profiles
def process_run_data(run_data, equiv_grad_list, success_list, f_sol, accuracy):
    f0 = float(run_data[0]['f_vals'])  # first loss value
    success_grad_evals = None

    # Read row by row and compute normalised_loss
    for row in run_data:
        normalised_loss = (float(row['f_vals']) - f_sol) / (f0 - f_sol)
        if normalised_loss <= accuracy:
            success_grad_evals = float(row['equiv_grad_evals'])
            break

    # Check if the largest element in equiv_grad_list is larger than the largest element in run_data['equiv_grad_evals']
    max_equiv_grad_list = max(equiv_grad_list)
    max_run_data_equiv_grad_evals = max(float(row['equiv_grad_evals']) for row in run_data)
    if max_equiv_grad_list > max_run_data_equiv_grad_evals:
        raise ValueError("Placeholder error message: equiv_grad_list has larger values than run_data equiv_grad_evals")

    # If a success is found, update success_list
    if success_grad_evals is not None:
        success_grad_index = next((i for i, val in enumerate(equiv_grad_list) if val >= success_grad_evals), None)
        
        if success_grad_index is not None:
            for j in range(success_grad_index, len(success_list)):
                success_list[j] += 1

