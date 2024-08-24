import scipy, scipy.optimize, scipy.linalg
import autograd.numpy as np
import json
import re

class SolverOutput():
    def __init__(self, solver, final_f_val=None, final_x=None, final_k=None, f_vals=None, update_norms=None, **kwargs):
        self.solver = solver
        self.final_f_val = final_f_val
        self.final_x = final_x
        self.final_k = final_k
        self.f_vals = f_vals
        self.update_norms = update_norms # stores the norms of all the iterate update vectors

        # Any other data structures we choosw to feed to __init__
        for key, value in kwargs.items():
            setattr(self, key, value)
        
class SolverOutputAverage():
    def __init__(self, solver, **kwargs):
        self.solver = solver

        # Some suggestions for relevant quantities to average:
        # final k
        # final derivs evaluated
        # final loss value
        # final gradient norm
        # ...
        
        # Data structures we choose to feed to __init__
        for key, value in kwargs.items():
            setattr(self, key, value)

def update_best_known_result(extended_objective_name, new_best_value):
    best_known_results_file_name = 'results/best_known_results.json'
    with open(best_known_results_file_name, 'r') as f:
        best_known_results = json.load(f)
    
    if extended_objective_name not in best_known_results or new_best_value < best_known_results[extended_objective_name]:
        best_known_results[extended_objective_name] = new_best_value
        with open(best_known_results_file_name, 'w') as f:
            json.dump(best_known_results, f, indent=4)

def problem_name_dim_tuple_from_json_name(json_prob_name: str):
    match = re.match(r"([A-Za-z0-9_\-]+)_n(\d+)", json_prob_name)
    if match:
        return match.group(1), int(match.group(2))
    else:
        raise ValueError("String format is not as expected")


def normalise_loss(loss_data: list, f_sol: float, f0: float):
    """
    Input arguments:
    loss_data: list containing loss values.
    f_sol: Minimimum of the problem.
    f0: Function value at the starting point x0 of the algorithm run.
    """
    out = [((loss - f_sol) / (f0 - f_sol)) for loss in loss_data]
    return out

# This function takes in a problem, a list of solver objects, and an
# integer specifying how many times each solver should be run on the problem.
# The output argument is a list of tuples.
def average_solver_runs(problem: tuple, solver_list: list,
                          no_runs: int, result_attrs: list):
    """
    INPUT arguments:
    problem: tuple (x0, Objective); output of one of the functions in /problems/test_problems.py
    solver_list:  list of solver objects to be used in runs
    no_runs:      number of solver runs to be performed per item in solver_list
    result_attrs: list of strings. Each string should be the name of a
    suitable SolverOutput attribute whose average should be computed. 

    OUTPUT argument:
    output_dict: a dictionary with THREE key-value pairs.
    It is of the form
    {
    'problem': (x0, Objective),
    'no_runs': no_runs,
    'avg_results': [(Solver, SolverOutputAverage), ..., (Solver, SolverOutputAverage)]
    }
    """

    output_dict = {'problem': problem, 'no_runs': no_runs,
                   'avg_results': [], 'raw_results': []}
    x0, obj = problem # unpack for ease of use
    for solver in solver_list:
        if solver.obj.name != obj.name:
            raise Exception('Problem and solvers inputs have different problems specified!')
        results_dict = {attr_name: [] for attr_name in result_attrs}
        raw_solver_outputs = []
        for i in range(no_runs):
            solver_output = solver.optimise(x0)
            raw_solver_outputs.append(solver_output)
            for attr_name in results_dict:
                results_dict[attr_name].append(getattr(solver_output, attr_name))
        
        # Calculate averages
        avg_results = {f"{attr_name}_avg": np.mean(results_dict[attr_name]) for attr_name in results_dict}

        # Create a SolverOutputAverage instance with the averaged results
        solver_avg = SolverOutputAverage(solver, **avg_results)
        
        # Append to the output list, INCLUDING RAW results
        output_dict['avg_results'].append((solver, solver_avg))
        output_dict['raw_results'].append((solver, raw_solver_outputs))
    
    return output_dict



# Can use scipy's implementation of a strong-Wolfe-condition-ensuring
# linesearch (provided direction is a descent direction, of course).
# We use this in (L)BFGS linesearch methods to ensure positive-definiteness of
# hessian approximations.
def strong_wolfe_linesearch(func, grad_func, x, direction, c1, c2, max_iter):
    return scipy.optimize.line_search(func, grad_func, x,
                                        direction, c1=c1, c2=c2,
                                        maxiter=max_iter)


# TALL(!) scaled Gaussian sketching matrix!
# Scaling as  used, e.g., in cartis2022 paper
# Subspace dimension taken to be the number of columns, n
def scaled_gaussian(m, n):
    return np.random.normal(scale=np.sqrt(1 / n), size=(m, n))

# NOTE: The function below is adapted from Mezzadri2007, Sec. 5
# (arXiv:math-ph/0609050)
# Generates a random orthonormal matrix with dimensions (m, n)
def haar(m, n):
    z = np.random.randn(m, n) / np.sqrt(2.0)
    q,r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q, ph, q)

    return q

# The below is *inspired by* Algorithm 5 from https://doi.org/10.1007/s10107-022-01836-1
# It returns a (potentially orthonormal) matrix including the directions of curr_mat's columns
# along with no_dirs (int) random directions.
def append_dirs(curr_mat: np.ndarray, ambient_dim: int,
                no_dirs: int, curr_is_orth: bool,
                orthogonalise: bool,
                normalise_cols: bool):
    if not (orthogonalise or normalise_cols):
        raise Exception('Pretty sure you would NOT want to do this without orthogonalising or at least normalising the columns of P_k.')

    # If 0 directions to be added, simply return the
    # (maybe orthogonalised) input matrix.
    if no_dirs == 0:
        if curr_is_orth or (not orthogonalise):
            return curr_mat
        else:
            curr_mat_orth, _ = np.linalg.qr(curr_mat)
            return curr_mat_orth


    n = ambient_dim # column/ambient dimension
    A = np.random.randn(n, no_dirs) # draw (scaled!) Gaussian matrix

    if curr_mat is None:
        stack = A
    else:
        stack = np.hstack((curr_mat, A))
    
    if orthogonalise:
        output, _ = np.linalg.qr(stack)
    else:
        if normalise_cols: # only relevant if not orthogonalising already
            output = normalise_matrix_columns(stack)
        else: # raw
            output = stack
    
    return output

# This matrix, in line with our proposed "unified" definition of a true iteration,
# returns a copy of the input matrix where each of its columns are normalised to
# have unit Euclidian norm.
def normalise_matrix_columns(mat: np.ndarray) -> np.ndarray:
    # Copy the input matrix
    mat_copy = mat.copy()
    
    # Iterate over each column to normalise
    for i in range(mat_copy.shape[1]):
        col_norm = np.linalg.norm(mat_copy[:, i])
        if col_norm == 0: # usually because an update vector was very very small,
            # numerically may as well essentially have been random. 
            temp = np.random.randn(mat_copy.shape[0])
            mat_copy[:, i] = temp / (np.linalg.norm(temp))
        else:
            mat_copy[:, i] /= col_norm
    
    return mat_copy

# TODO: IMPLEMENT LSR1 updates given a B0 and a matrix storing 
# secant pairs which were already ensured to be suitable! 
# Compact L-SR1 representation taken from Theorem 9.1,
# Nocedal & Wright, 1st ed., §9.2.
def lsr1(B0, Y, X):
    if Y is None or X is None:
        if not (X is None and Y is None):
            raise Exception('One of X and Y is None, but not both of them!')
        else:
            return B0
    k = Y.shape[1]
    if X.shape[1] != k:
        raise Exception(f'Number of columns in X and in Y should match! We have {X.shape[1]} in X and {Y.shape[1]} in Y.')
    
    # NOTE: again, using terminology of Nocedal & Wright.
    proto_L = np.dot(X.T, Y)
    L = np.tril(proto_L, k=-1)
    D = np.diag(np.diag(proto_L))

    mat_one = Y - np.dot(B0, X)
    mat_two = D + L + L.T - np.dot(X.T, np.dot(B0, X))
    to_add = np.dot(mat_one, np.dot(np.linalg.inv(mat_two) , mat_one.T))
    B = B0 + to_add
    return B
    

