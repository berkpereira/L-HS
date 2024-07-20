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

# NOTE: this functions keeps only KEEP_NO solver runs' data.
# These are prioritised vs any incoming data by the equivalent gradient
# evaluations budget of the run. The larger the budget, the higher the priority.
# NOTE: KEEP_NO is hard-coded here! For instance, Cartis' and Roberts' 2023
# paper uses KEEP_NO equal to 10.
def save_solver_output(problem_name: str, solver_config_str: str,
                       solver_output: SolverOutput, output_dir='results'):
    
    KEEP_NO = 10
    
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
        'proj_grad_norms': solver_output.proj_grad_norms
    }
    
    # Load existing data if the file exists
    existing_data = []
    if os.path.isfile(file_path):
        with open(file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                for key in ['equiv_grad_budget', 'iter',
                            'deriv_evals', 'equiv_grad_evals', 'f_vals',
                            'update_norms', 'full_grad_norms', 'proj_grad_norms']:
                    if row[key] != '':
                        row[key] = eval(row[key])
                    else:
                        row[key] = ''
                existing_data.append(row)

    # Group data by run_id and get the maximum equiv_grad_budget for each run
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
        })
    grouped_data[data['run_id']] = run_data
    
    # Sort runs by equiv_grad_budget and keep only the top KEEP_NO
    sorted_runs = sorted(grouped_data.values(), key=lambda x: float(x[0]['equiv_grad_budget']), reverse=True)[:KEEP_NO]

    # Write the selected runs back to the file
    with open(file_path, mode='w', newline='') as file:
        fieldnames = ['run_id', 'equiv_grad_budget', 'iter', 'deriv_evals', 'equiv_grad_evals', 'f_vals', 'update_norms', 'full_grad_norms', 'proj_grad_norms']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in sorted_runs:
            for row in run:
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
