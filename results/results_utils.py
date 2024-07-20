import json
import os
import csv
import hashlib
from datetime import datetime
from solvers.utils import SolverOutput

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

def save_solver_output(problem_name: str, solver_config_str: str,
                       solver_output: SolverOutput, output_dir='results'):
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
        'grad_evals': (solver_output.deriv_evals / solver_output.solver.obj.input_dim),
        'f_vals': solver_output.f_vals,
        'update_norms': solver_output.update_norms,
        'full_grad_norms': solver_output.full_grad_norms,
        'proj_grad_norms': solver_output.proj_grad_norms
    }
    
    # If the file does not exist, create it and write the header
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        fieldnames = ['run_id', 'iter', 'deriv_evals', 'grad_evals', 'f_vals', 'update_norms', 'full_grad_norms', 'proj_grad_norms']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for iter in range(len(solver_output.f_vals)):
            row = {
                'run_id': data['run_id'],
                'iter': iter,
                'deriv_evals': data['deriv_evals'][iter] if data['deriv_evals'].size > iter else None,
                'grad_evals':  data['grad_evals'][iter] if data['grad_evals'].size > iter else None,
                'f_vals': data['f_vals'][iter],
                'update_norms': data['update_norms'][iter] if data['update_norms'].size > iter else None,
                'full_grad_norms': data['full_grad_norms'][iter] if data['full_grad_norms'].size > iter else None,                'proj_grad_norms': data['proj_grad_norms'][iter] if data['proj_grad_norms'].size > iter else None,
            }
            writer.writerow(row)

def load_solver_results(problem_name: str, solver_config, output_dir='results'):
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
    file_path = os.path.join(output_dir, problem_name, f'{hashed_filename}.csv')

    if not os.path.exists(file_path):
        raise Exception('Result file does not exist.')

    # Load the results from the CSV file
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        results = [row for row in reader]

    return results
